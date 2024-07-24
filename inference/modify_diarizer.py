import site
import os

site_packages_path = site.getsitepackages()[0]
file_path = os.path.join(site_packages_path, 'nemo', 'collections', 'asr', 'models', 'clustering_diarizer.py')

modifications = """
def _extract_embeddings(self, manifest_file: str, scale_idx: int, num_scales: int):
    print("INSIDE MODIFIED EXTRACT EMBEDDINGS3..")
    import nemo.collections.asr as nemo_asr
    self._ecapa_tdnn_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='ecapa_tdnn')
    self._ecapa_tdnn_model.eval()

    logging.info("Extracting embeddings for Diarization")
    self._setup_spkr_test_data(manifest_file)
    self.embeddings = {}
    self._speaker_model.eval()
    self.time_stamps = {}

    all_embs = torch.empty([0])
    for test_batch in tqdm(
        self._speaker_model.test_dataloader(),
        desc=f'[{scale_idx+1}/{num_scales}] extract embeddings',
        leave=True,
        disable=not self.verbose,
    ):
        test_batch = [x.to(self._speaker_model.device) for x in test_batch]
        audio_signal, audio_signal_len, labels, slices = test_batch
        with autocast():
            _, embs = self._speaker_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
            embs_ecapa = self._ecapa_tdnn_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
            embs = torch.cat((embs, embs_ecapa[0]), dim=-1)
            emb_shape = embs.shape[-1]
            embs = embs.view(-1, emb_shape)
            all_embs = torch.cat((all_embs, embs.cpu().detach()), dim=0)
        del test_batch

    with open(manifest_file, 'r', encoding='utf-8') as manifest:
        for i, line in enumerate(manifest.readlines()):
            line = line.strip()
            dic = json.loads(line)
            uniq_name = get_uniqname_from_filepath(dic['audio_filepath'])
            if uniq_name in self.embeddings:
                self.embeddings[uniq_name] = torch.cat((self.embeddings[uniq_name], all_embs[i].view(1, -1)))
            else:
                self.embeddings[uniq_name] = all_embs[i].view(1, -1)
            if uniq_name not in self.time_stamps:
                self.time_stamps[uniq_name] = []
            start = dic['offset']
            end = start + dic['duration']
            self.time_stamps[uniq_name].append([start, end])

    if self._speaker_params.save_embeddings:
        embedding_dir = os.path.join(self._speaker_dir, 'embeddings')
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir, exist_ok=True)

        prefix = get_uniqname_from_filepath(manifest_file)
        name = os.path.join(embedding_dir, prefix)
        self._embeddings_file = name + f'_embeddings.pkl'
        pkl.dump(self.embeddings, open(self._embeddings_file, 'wb'))
        logging.info("Saved embedding files to {}".format(embedding_dir))
"""

if os.path.isfile(file_path):
    with open(file_path, 'r') as file:
        file_data = file.read()

    start_idx = file_data.find("def _extract_embeddings")
    end_idx = file_data.find("def ", start_idx + 1) if file_data.find("def ", start_idx + 1) != -1 else len(file_data)

    indent = " " * 4
    indented_modifications = "\n".join([indent + line if line.strip() else line for line in modifications.split('\n')])

    new_file_data = file_data[:start_idx] + indented_modifications + "\n\n" + indent + file_data[end_idx:]

    with open(file_path, 'w') as file:
        file.write(new_file_data)

    print(f"Modified function in {file_path}.")
else:
    print(f"File not found: {file_path}")
