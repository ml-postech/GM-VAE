from .dataset import add_dataset_args
from .models import add_model_args

def add_task_args(parser):
    group = parser.add_argument_group('CelebA')
    add_dataset_args(group)
    add_model_args(group)

