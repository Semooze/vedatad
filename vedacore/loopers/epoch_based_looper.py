from .base_looper import BaseLooper
import numpy as np

class EpochBasedLooper(BaseLooper):

    def __init__(self, modes, dataloaders, engines, hook_pool, logger,
                 workdir):
        super().__init__(modes, dataloaders, engines, hook_pool, logger,
                         workdir)

    def epoch_loop(self, mode):
        self.mode = mode
        dataloader = self.dataloaders[mode]

        engine = self.engines[mode]
        for idx, data in enumerate(dataloader):
            print('*'*80)
      
            print(idx, data['video_metas'].data[0][0]['ori_video_name'])
            print(type(data['imgs'].data[0].cpu().numpy().shape))
            print(data['imgs'].data[0].cpu().numpy().shape)
            # print(type(data['imgs'].data))
            # print(len(data['imgs'].data))
            # print(data['imgs'].data[0].cpu().numpy().shape)

            # with open(f"/home/jupyter/project/new_data/train/{data['video_metas'].data[0][0]['ori_video_name']}.npy", 'wb') as f:
            #     np.save(f, data['imgs'].data[0].cpu().numpy())
            # print(data['video_metas'])
            # print('*'*80)
            # print(data['imgs'])
            self.hook_pool.fire(f'before_{mode}_iter', self)
            result = engine(data)
            print(result)
            self.cur_results[mode] = result
            continue
            if mode == BaseLooper.TRAIN:
                self._iter += 1
            self._inner_iter = idx + 1
            self.hook_pool.fire(f'after_{mode}_iter', self)
        raise Exception
    def start(self, max_epochs):
        self.hook_pool.fire('before_run', self)
        while self.epoch < max_epochs:
            for mode in self.modes:
                mode = mode.lower()
                self.hook_pool.fire(f'before_{mode}_epoch', self)
                self.epoch_loop(mode)
                if mode == BaseLooper.TRAIN:
                    self._epoch += 1
                self.hook_pool.fire(f'after_{mode}_epoch', self)
            if len(self.modes) == 1 and self.modes[0] == EpochBasedLooper.VAL:
                break
