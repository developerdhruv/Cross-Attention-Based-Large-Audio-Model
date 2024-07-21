from msclap import CLAP

class ClapEncoder:
    def __init__(self, model_version='2023', use_cuda=False):
        self.model = CLAP(version=model_version, use_cuda=use_cuda)
    def get_text_embeddings(self, texts):
        return self.model.get_text_embeddings(texts)
    def get_audio_embeddings(self, audio_paths):
        return self.model.get_audio_embeddings(audio_paths)  