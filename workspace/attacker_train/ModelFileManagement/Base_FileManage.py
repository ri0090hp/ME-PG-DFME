from pathlib import Path

class FileManagement():
    def __init__(self, file_name, model_size):
        self.file_name = file_name
        self.model_size = model_size
        self.cloud_model_dir = Path("/workspace/cloud_model/checkpoints_cloud")
        self.checkpoints_student_dir = Path("/workspace/attacker_train/checkpoints_student")

    def load_targetmodel(self):
        pass

    def get_studentmodel(self):
        pass

    def normal_save(self):
        pass
    