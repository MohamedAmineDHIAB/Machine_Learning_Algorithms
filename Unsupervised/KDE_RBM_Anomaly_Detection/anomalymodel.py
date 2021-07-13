import sklearn,sklearn.metrics
import numpy as np
import utils
Xr,Xi,Xo = utils.getdata()

class AnomalyModel:

    def auroc(self):
        Ei = self.energy(Xi)
        Eo = self.energy(Xo)
        return sklearn.metrics.roc_auc_score(
            np.concatenate([Ei*0+0,Eo*0+1]),
            np.concatenate([Ei,Eo])
        )