import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

from models.SSNet import SSNet

labels = ['TC','EC','FS','WJ','Snow','Ocean','Desert','Veg','Ci','Cs','DC','AC','AS','Ns','Cu','Sc','St']

def load_saved_model(saved_model_name,model):
    print(saved_model_name,123)
    model_path = '../results/'+ saved_model_name+'/best_model.pt'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def direct_load_emb(saved_model_name):
    model_path = '../results/' + saved_model_name + '/best_model.pt'
    checkpoint = torch.load(model_path)
    emb = checkpoint['state_dict']['label_emb.weight']
    return emb

def draw_tsine(emb,dir:str,name:str):
    ##data = np.random.rand(17, 10)  # 64个样本，每个样本维度为10
    data = emb
    target = np.arange(17)  # 生成64个标签，用于区分样本目标
    t_sne_features = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=7).fit_transform(data)
    plt.scatter(x=t_sne_features[:, 0], y=t_sne_features[:, 1], c=target, cmap='jet')
    # plt.axes.get_xaxis().set_visible(False)  # 隐藏x坐标轴
    # plt.axes.get_yaxis().set_visible(False)  # 隐藏y坐标轴
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')
    for i in range(17):
        plt.annotate(labels[i], xy=(t_sne_features[i, 0], t_sne_features[i, 1]),
                     xytext=(t_sne_features[i, 0] + 1, t_sne_features[i, 1] + 1))
    plt.savefig(dir+name+'.png',dpi=300)
    plt.show()

if __name__ == '__main__':
    model = SSNet(16, 17)
    res_dir = 's2net.LSCIDMR_16c.bsz_64.adam5e-05.1_1_spat_layers.1_1_spec_layers.1overall_layers.dropout01.plateau_on_map.vggPre_vggLeaky_scnLeaky_transGelu_7533_111_alpha5_spec_bug_solve_emsamble_grad_patience3_voteweight0.1_unit_matrix'
    #model = load_saved_model(res_dir,model)
    # emb = model.label_emb.weight.detach()
    # emb = model.label_emb_map(emb).cpu().detach().numpy()
    emb = direct_load_emb(res_dir).cpu().detach().numpy()
    print(emb.shape)
    draw_tsine(emb,'../fig/query_scatter/','no_mlp')