import matplotlib.pyplot as plt

default_path = './results/LSCIDMR.3layer.bsz_32.sgd5e-05.geo_emb/test.log'
def get_data_seq(path:str=default_path,indicator:str='mAP:'):
    file = open(path)
    lines = file.readlines(100000)
    #print(lines)
    seq = []

    for line in lines:
        start = line.find(indicator)
        if start >= 0 :
            start += len(indicator)
            num = float(line[start:start+5])
            seq.append(num)
    return seq

if __name__ == '__main__':
    bbv_path = '../results/s2net.LSCIDMR_16c.bsz_64.adam5e-05.1_1_spat_layers.1_1_spec_layers.1overall_layers.dropout00.plateau_on_map.vggPre_vggLeaky_scnLeaky_transLeaky_7533_111_alpha10_mlp_grad_not_group_bbv/test.log'
    mean_path = '../results/s2net.LSCIDMR_16c.bsz_64.adam5e-05.1_1_spat_layers.1_1_spec_layers.1overall_layers.dropout00.plateau_on_map.vggPre_vggLeaky_scnLeaky_transLeaky_7533_111_alpha10_mlp_not_group_mean/test.log'
    no_punish_path = '../results/s2net.LSCIDMR_16c.bsz_64.adam5e-05.1_1_spat_layers.1_1_spec_layers.1overall_layers.dropout01.plateau_on_map.vggPre_vggLeaky_scnLeaky_transLeaky_7533_111_alpha10_mlp_not_group_simpleVote/test.log'

    metric = 'mAP:'
    bbv_seq = get_data_seq(bbv_path, metric)[1:101]
    mean_seq = get_data_seq(mean_path, metric)[1:101]
    no_punish_seq = get_data_seq(no_punish_path, metric)[1:101]


    print('123'.find('1'))
    plt.figure(figsize=(7,6.7))
    w = 1
    textsize = 20
    plt.plot(range(int(len(mean_seq))), mean_seq, '#55AA55', lw=w, label='mean')
    plt.plot(range(int(len(no_punish_seq))), no_punish_seq, '#5555AA', lw=w, label='simple vote')
    plt.plot(range(int(len(bbv_seq))), bbv_seq, '#AA5555', lw=w, label='BBV')
    label_font = {'family':'DejaVu Sans','weight':'normal','size':textsize}
    plt.xlabel('epoch',label_font)
    plt.ylabel(metric[:-1], label_font)
    plt.tick_params(labelsize=12.5)
    #plt.ylim(4,24)
    #plt.ylim(ylim_min, ylim_max)
    plt.legend(prop={'size':textsize})
    plt.savefig('../fig/converge/converge_'+metric[:-1]+'.png', dpi=600)
    plt.show()