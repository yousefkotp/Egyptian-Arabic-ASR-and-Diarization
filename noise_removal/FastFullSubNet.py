import torch
import torch.nn as nn
import torchaudio as audio
from torch.nn import functional

from noise_removal.utils import FullSubNetBaseModel, SequenceModel, stft, istft, decompress_cIRM


class FastFullSubNet(FullSubNetBaseModel):
    def __init__(
        self,
        look_ahead=2,
        shrink_size=2,
        sequence_model="LSTM",
        num_mels=64,
        encoder_input_size=257,
        bottleneck_hidden_size=384,
        bottleneck_num_layers=2,
        noisy_input_num_neighbors=5,
        encoder_output_num_neighbors=0,
        norm_type="offline_laplace_norm",
        weight_init=False,
        sr=16000,
        n_ftt=512,
        win_length=512,
        hop_length=256,
    ):
        """Fast FullSubNet.

        Notes:
            Here, the encoder, bottleneck, and decoder are corresponding to the F_l2m, S, and F_m2l models in the paper, respectively.
        """
        super().__init__()
        assert sequence_model in ("GRU", "LSTM"), f"{self.__class__.__name__} only support GRU and LSTM."

        # F_l2m
        self.encoder = nn.Sequential(
            SequenceModel(
                input_size=64,
                hidden_size=384,
                output_size=0,
                num_layers=1,
                bidirectional=False,
                sequence_model=sequence_model,
                output_activate_function=None
            ),
            SequenceModel(
                input_size=384,
                hidden_size=257,
                output_size=64,
                num_layers=1,
                bidirectional=False,
                sequence_model=sequence_model,
                output_activate_function="ReLU"
            ),
        )

        # Mel filtering
        self.mel_scale = audio.transforms.MelScale(
            n_mels=num_mels,
            sample_rate=16000,
            f_min=0,
            f_max=8000,
            n_stft=encoder_input_size,
        )

        # S
        self.bottleneck = SequenceModel(
            input_size=(noisy_input_num_neighbors * 2 + 1) + (encoder_output_num_neighbors * 2 + 1),
            output_size=1,
            hidden_size=bottleneck_hidden_size,
            num_layers=bottleneck_num_layers,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function="ReLU"
        )

        # F_m2l
        self.decoder_lstm = nn.Sequential(
            SequenceModel(
                input_size=64 + 64,
                hidden_size=512,
                output_size=0,
                num_layers=1,
                bidirectional=False,
                sequence_model=sequence_model,
                output_activate_function=None
            ),
            SequenceModel(
                input_size=512,
                hidden_size=512,
                output_size=257 * 2,
                num_layers=1,
                bidirectional=False,
                sequence_model=sequence_model,
                output_activate_function=None,
            ),
        )

        self.shrink_size = shrink_size
        self.look_ahead = look_ahead
        self.num_mels = num_mels
        self.noisy_input_num_neighbors = noisy_input_num_neighbors
        self.enc_output_num_neighbors = encoder_output_num_neighbors
        self.norm = self.norm_wrapper(norm_type)
        
        self.sr = sr
        self.n_ftt = n_ftt
        self.hop_length = hop_length
        self.win_length = win_length
        
        self.stft_args = {
            "n_fft": self.n_ftt,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
        }

        if weight_init:
            self.apply(self.weight_init)

    def real_time_downsampling(self, input):
        """Downsampling an input tensor long time.

        Args:
            input: tensor with the shape of [B, C, F, T].

        Returns:
            Donwsampled tensor with the shape of [B, C, F, T // shrink_size].
        """
        first_block = input[..., 0:1]  # [B, C, F, 1]
        block_list = torch.split(input[..., 1:], self.shrink_size, dim=-1)  # ([B, C, F, shrink_size], [B, C, F, shrink_size], ...)
        last_block = block_list[-1]  # [B, C, F, T]

        output = torch.cat(
            (
                first_block,  # [B, C, F, 1]
                torch.mean(torch.stack(block_list[:-1], dim=-1), dim=-2),  # [B, C, F, T]
                torch.mean(last_block, dim=-1, keepdim=True)  # [B, C, F, 1]
            ), dim=-1
        )  # [B, C, F, T // shrink_size]

        return output

    def real_time_upsampling(self, input, target_len=False):
        *_, n_frames = input.shape
        input = input[..., None]  # [B, C, F, T, 1]
        input = input.expand(*input.shape[:-1], self.shrink_size)  # [B, C, F, T, shrink_size]
        input = input.reshape(*input.shape[:-2], n_frames * self.shrink_size)  # [B, C, F, T * shrink_size]

        if target_len:
            input = input[..., :target_len]

        return input

    def full_band_crm_mask(self, pred_crm: torch.Tensor, noisy: torch.Tensor, noisy_real, noisy_imag):
        pred_crm = decompress_cIRM(pred_crm)
        enhanced_real = pred_crm[..., 0] * noisy_real - pred_crm[..., 1] * noisy_imag
        enhanced_imag = pred_crm[..., 1] * noisy_real + pred_crm[..., 0] * noisy_imag
        enhanced: torch.Tensor = istft(
            (enhanced_real, enhanced_imag), length=noisy.size(-1), input_type="real_imag", **self.stft_args
        )
        if enhanced.dim() == 3:
            enhanced = enhanced.squeeze(0)
        return enhanced
    
    # fmt: off
    def forward(self, data):
        """Forward pass.

        Args:
            data: dictionary containing the noisy waveform.

        Returns:
            The real part and imag part of the enhanced spectrogram with shape [B, 2, F, T].

        Notes:
            noisy: noisy waveform
            mix_mag: noisy magnitude spectrogram with shape [B, 1, F, T].
            B - batch size
            C - channel
            F - frequency
            F_mel - mel frequency
            T - time
            F_s - sub-band frequency
        """
        noisy = data['noisy']
        noisy = noisy.squeeze(1)
        mix_mag, _, noisy_real, noisy_imag = stft(noisy, **self.stft_args)
        mix_mag = mix_mag.unsqueeze(1)
        assert mix_mag.dim() == 4
        mix_mag = functional.pad(mix_mag, [0, self.look_ahead])  # Pad the look ahead
        batch_size, num_channels, num_freqs, num_frames = mix_mag.size()
        assert num_channels == 1, f"{self.__class__.__name__} takes a magnitude feature as the input."

        # Mel filtering
        mix_mel_mag = self.mel_scale(mix_mag)  # [B, C, F_mel, T]
        _, _, num_freqs_mel, _ = mix_mel_mag.shape

        # F_l2m
        enc_input = self.norm(mix_mel_mag).reshape(batch_size, -1, num_frames)
        enc_output = self.encoder(enc_input).reshape(batch_size, num_channels, -1, num_frames)  # [B, C, F, T]

        # Unfold - noisy spectrogram, [B, N=F, C, F_s, T]
        mix_mel_unfold_mag = self.freq_unfold(mix_mel_mag, num_neighbors=self.noisy_input_num_neighbors)  # [B, F_mel, C, F_sub, T]
        mix_mel_unfold_mag = mix_mel_unfold_mag.reshape(batch_size, self.num_mels, self.noisy_input_num_neighbors * 2 + 1, num_frames)  # [B, F_mel, F_sub, T]

        # Unfold - full-band model's output, [B, N=F, C, F_f, T], where N is the number of sub-band units
        enc_output_unfold_mel = self.freq_unfold(enc_output, num_neighbors=self.enc_output_num_neighbors)  # [B, F_mel, C, F_sub, T]
        enc_output_unfold_mel = enc_output_unfold_mel.reshape(batch_size, self.num_mels, self.enc_output_num_neighbors * 2 + 1, num_frames)  # [B, F_mel, F_sub, T]

        # Bottleneck (S)
        bn_input = torch.cat([mix_mel_unfold_mag, enc_output_unfold_mel], dim=2)
        num_sb_unit_freqs = bn_input.shape[2]

        # Bottleneck - time downsampling
        bn_input_shrink = self.real_time_downsampling(bn_input)  # [B, F_mel, F_sub_1 + F_sub_2, T // shrink_size]
        bn_input_shrink = self.norm(bn_input_shrink)  # [B, F_mel, F_sub_1 + F_sub_2, T // shrink_size]
        bn_input_shrink = bn_input_shrink.reshape(batch_size * self.num_mels, num_sb_unit_freqs, -1)  # [B * F_mel, F_sub_1 + F_sub_2, T // shrink_size]
        bn_output_shrink = self.bottleneck(bn_input_shrink)  # [B * F_mel, 1, T // shrink_size]
        bn_output_shrink = bn_output_shrink.reshape(batch_size, self.num_mels, 1, -1).permute(0, 2, 1, 3)  # [B, 1, F_mel, T // shrink_size]
        bn_output = self.real_time_upsampling(bn_output_shrink, target_len=num_frames)  # [B, 1, F_mel, T]

        # F_ml2
        dec_input = torch.cat([enc_output, bn_output], dim=2)
        dec_input = dec_input.reshape(batch_size, -1, num_frames)
        decoder_lstm_output = self.decoder_lstm(dec_input)  # [B * C, F * 2, T]
        dec_output = decoder_lstm_output.reshape(batch_size, 2, num_freqs, num_frames)

        # Output
        output = dec_output[:, :, :, self.look_ahead:]
        output = output.permute(0, 2, 3, 1)
        
        # Full band CRM mask
        enhanced = self.full_band_crm_mask(output, noisy, noisy_real, noisy_imag)
        enhanced = enhanced.unsqueeze(1)
        return enhanced, output