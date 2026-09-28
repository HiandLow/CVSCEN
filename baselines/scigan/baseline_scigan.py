import sys
import os
import numpy as np

class SCIGANWrapper:
    def __init__(self, num_features, **kwargs):
        import os
        os.environ['TF_USE_LEGACY_KERAS'] = '1'
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        self.num_features = num_features
        self.hparams = kwargs
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:

            sys.path.insert(0, current_dir)
            
        vscen_utils = sys.modules.pop('utils', None)
        from SCIGAN import SCIGAN_Model
        if vscen_utils is not None:
            sys.modules['utils'] = vscen_utils
            
        self.SCIGAN_Model_Class = SCIGAN_Model
        
    def fit(self, X, T, Y):
        import os
        os.environ['TF_USE_LEGACY_KERAS'] = '1'
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        import numpy as np
        
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(int(np.random.randint(2**31 - 1)))
        
        dim = self.hparams.get('dim_layer', 128)
        params = {
            'num_features': self.num_features,
            'num_treatments': 1,
            'num_dosage_samples': self.hparams.get('num_dosage_samples', 5),
            'export_dir': '',
            'alpha': self.hparams.get('alpha', 1.0),
            'batch_size': self.hparams.get('batch_size', 16),
            'h_dim': dim,
            'h_inv_eqv_dim': dim,
            'lr_g': self.hparams.get('lr_g', 0.001),
            'lr_d': self.hparams.get('lr_d', 0.001),
            'd_steps': self.hparams.get('d_steps', 1)
        }
        
        params['iterations_gan'] = self.hparams.get('iterations_gan', 5000)
        params['iterations_inf'] = self.hparams.get('iterations_inf', 10000)
        
        self.model = self.SCIGAN_Model_Class(params)
        
        Train_T = np.zeros(X.shape[0], dtype=int)
        Train_D = T
        self.model.train(Train_X=X, Train_T=Train_T, Train_D=Train_D, Train_Y=Y, verbose=False)
        
    def predict(self, X, T):
        import os
        os.environ['TF_USE_LEGACY_KERAS'] = '1'
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        
        n = X.shape[0]
        if np.isscalar(T):
            T = np.full(n, T)
            
        num_dosage_samples = self.hparams.get('num_dosage_samples', 5)
        treatment_dosage_samples = np.zeros([n, 1, num_dosage_samples])
        for i in range(n):
            treatment_dosage_samples[i, 0, 0] = T[i]
            
        I_logits = self.model.sess.run(
            self.model.sess.graph.get_tensor_by_name("inference_outcomes:0"),
            feed_dict={
                self.model.X: X,
                self.model.Treatment_Dosage_Samples: treatment_dosage_samples
            }
        )
        pred = I_logits[:, 0, 0]
        return pred
