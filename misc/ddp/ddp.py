#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: atimans
"""

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


model.to(rank)
ddp_model = DDP(model, device_ids=[rank])
optimizer = optim.Adam(ddp_model.parameters())

X.to(rank), y.to(rank)
out = ddp_model(X)

dist.destroy_process_group()


main():
    import torch.multiprocessing as mp
    world_size = 2
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)


# checkpointing
if rank == 0:
     torch.save(ddp_model.state_dict())

dist.barrier() # block other processes from loading while saving
map_location={} #needs to be properly configured



### Data Parallel
https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html

model = nn.DataParallel(model)


###
main:
    args(-nr_gpus)
    get_device
    if nr_gpus > 1 and data_parallel:
        world_size = nr_gpus
        mp.spawn(run_model_ddp,
            args=(world_size,),
            nprocs=world_size,
            join=True)
    else:
        run_model


    args(-nr_gpus)
    nr_gpus = args.nr_gpus
    get_device
    if nr_gpus > 1 and data_parallel:
        #os.environ["MASTER_ADDR"] = "localhost"
        #os.environ["MASTER_PORT"] = "8888"
        mp.spawn(run_model_ddp,
            args=(args,),
            nprocs=nr_gpus,
            join=True)
    else:
        run_model



def run_model_ddp(rank, args):
    dist.init_process_group("nccl", world_size=nr_gpus, rank=rank)
    run_model(rank, **args):
        torch.manual_seed(random_seed) #
        device = f"cuda:{rank}" # device = torch.device(device, rank)
        model.to(rank) # device
        ddp_model = DDP(model, device_ids=[rank])
        torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=nr_gpus, rank=rank,
                                                        shuffle=True)

        optimizer = optim.Adam(ddp_model.parameters())
        X.to(rank), y.to(rank)
        out = ddp_model(X)

        # for save checkpoints
        # if rank == 0: save one copy only since all models should be the same
            # save model/ checkpoint

    dist.destroy_process_group()


rank = int(os.environ.get("SLURM_NODEID"))
