import numpy as np
import torch
import pathlib
import shutil
from second.protos import pipeline_pb2
model_dir = "models"
model_dir = pathlib.Path(model_dir)
model_dir.mkdir(parents=True, exist_ok=True)
eval_checkpoint_dir = model_dir / 'eval_checkpoints'
eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
#if result_path is None:
 #   result_path = model_dir / 'results'
config_file_bkp = "pipeline.config"
config = pipeline_pb2.TrainEvalPipelineConfig()
#with open(config_path, "r") as f:
 #   proto_str = f.read()
 #   text_format.Merge(proto_str, config)
shutil.copyfile(config_path, str(model_dir / config_file_bkp))
input_cfg = config.train_input_reader
eval_input_cfg = config.eval_input_reader
model_cfg = config.model.second
train_cfg = config.train_config

class_names = list(input_cfg.class_names)
######################
# BUILD VOXEL GENERATOR
######################
voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
######################
# BUILD TARGET ASSIGNER
######################
bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
box_coder = box_coder_builder.build(model_cfg.box_coder)
target_assigner_cfg = model_cfg.target_assigner
target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                bv_range, box_coder)
######################
# BUILD NET
######################
center_limit_range = model_cfg.post_center_limit_range
model = second_builder.build(model_cfg, voxel_generator, target_assigner)
#net.cuda()

PATH = "models/voxelnet-12703.tckpt"
#model = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
#def computeTime(model, device='cuda'):
    #inputs = torch.randn(1, 3, 512, 1024)
    #if device == 'cuda':
      #  model = model.cuda()
     #   inputs = inputs.cuda()

    #model.eval()

    #i = 0
    #time_spent = []
   # while i < 100:
  #      start_time = time.time()
 #       with torch.no_grad():
#            _ = model(inputs)

      #  if device == 'cuda':
     #       torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
    #    if i != 0:
   #         time_spent.append(time.time() - start_time)
  #      i += 1
 #   print('Avg execution time (ms): {:.3f}'.format(np.mean(time_spent)))


#model = ESPNet()
#computeTime(model)
