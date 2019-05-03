import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator

from sklearn.manifold import TSNE

from prepare_data import save_in_pkl
from make_mismatched_pairs import read_in_pkl
from Doc2Vec_Modeler import Doc2VecModeler

def get_doc_ids(rt_id_lst, rt_doc_train_dict):
    return [rt_doc_train_dict[rt_id] for rt_id in rt_id_lst]

def get_pp_texts(doc_id_lst, train_pcorpus):
    return [train_pcorpus[doc_id].words for doc_id in doc_id_lst]

def get_inferred_vecs(dv_model, pp_txts):
    return [dv_model.model.infer_vector(pp_text, epochs=80).reshape(1, -1) for pp_text in pp_txts]

def combine_vecs(good_ref_vecs, good_tate_vecs, bad_ref_vecs, bad_tate_vecs):
    all_vecs_lst = []
    for idx, ref_vec in enumerate(good_ref_vecs):
        all_vecs_lst.append(ref_vec)
        all_vecs_lst.append(good_tate_vecs[idx])

    for idx, ref_vec in enumerate(bad_ref_vecs):
        all_vecs_lst.append(ref_vec)
        all_vecs_lst.append(bad_tate_vecs[idx])
    all_vecs_arr = np.concatenate(all_vecs_lst)
    return all_vecs_arr

def get_tsne_representations(all_vecs_arr, n_dim):
    X_embedded = TSNE(n_components=n_dim, random_state=42).fit_transform(all_vecs_arr)
    return X_embedded

def plot_2d_version(group_name, X_embedded_2, colors_dict, pair_labels_dict, save_fig=False):
    X = X_embedded_2[:,0]
    y = X_embedded_2[:,1]
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.set_title("Doc2Vec Representations of Annotations & Lyrics (2D)", fontsize=18)
    size = 200
    n_docs = X_embedded_2.shape[0]
    n_pairs = n_docs / 2
    for idx in range(n_docs):
        is_odd_idx = idx % 2 != 0
        alpha = 0.6 if is_odd_idx else 0.3
        ax.scatter(X[idx], y[idx], s=size, alpha=alpha, c=colors_dict[idx], label=pair_labels_dict[idx])
        if is_odd_idx:
            label_text = "Distance between {0} pair".format(pair_labels_dict[idx])
            line_format = ':' + colors_dict[idx][:1]
            plt.plot([X[idx-1], X[idx]], [y[idx-1], y[idx]], line_format, label=label_text);
    ax.legend(loc='upper center', ncol=n_pairs, markerscale=0.5, bbox_to_anchor=(0.5, -0.15))
    plt.show()
    if save_fig:
        file_name = '../images/{}_2d_tsne_plot.png'.format(groupname)
        fig.savefig(file_name)

def plot_3d_version(group_name, X_embedded_3, colors_dict, pair_labels_dict, save_fig=False):
    X = X_embedded_3[:,0]
    y = X_embedded_3[:,1]
    z = X_embedded_3[:,2]
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Doc2Vec Representations of Annotations & Lyrics (3D)", fontsize=18)
    size = 200
    n_docs = X_embedded_3.shape[0]
    n_pairs = n_docs / 2
    for idx in range(n_docs):
        is_odd_idx = idx % 2 != 0
        alpha = 0.6 if is_odd_idx else 0.3
        ax.scatter(X[idx], y[idx], z[idx], s=size, alpha=alpha, c=colors_dict[idx], label=pair_labels_dict[idx])
        if is_odd_idx:
            label_text = "Distance between {0} pair".format(pair_labels_dict[idx])
            line_format = ':' + colors_dict[idx][:1]
            plt.plot([X[idx-1], X[idx]], [y[idx-1], y[idx]], [z[idx-1], z[idx]], line_format, label=label_text);
    ax.legend(loc='upper center', ncol=int(n_pairs), markerscale=0.5, bbox_to_anchor=(0.5, -0.15))
    plt.show()
    if save_fig:
        file_name = '../images/{}_3d_tsne_plot.png'.format(groupname)
        fig.savefig(file_name)

def get_pairs_dist_info_df(dv_model, good_rtid, bad_rtid):
    tr_pair_df = rt_50.tr_pairings_df

    is_good_rtid = tr_pair_df[tr_pair_df['ref_id'].isin(good_rtid)]
    is_bad_rtid = tr_pair_df[tr_pair_df['ref_id'].isin(bad_rtid)]
    good_info_df = is_good_rtid[is_good_rtid['is_pair'] == 1][['ref_id', 'artist_name', 'ref_raw_text', 'tate_raw_text', 'pair_cs', 'pair_ed']]
    bad_info_df = is_bad_rtid[is_bad_rtid['is_pair'] == 1][['ref_id', 'artist_name', 'ref_raw_text', 'tate_raw_text', 'pair_cs', 'pair_ed']]
    all_info_df = pd.concat([good_info_df, bad_info_df])
    return all_info_df

def get_sorted_dist_info_df(all_info_df, use_cs=True):
    dist_col = 'pair_cs' if use_cs else 'pair_ed'
    sorted_df = all_info_df.sort_values([dist_col]).reset_index(drop=True)
    return sorted_df

def plot_cs_ed_3d_comparison(group_name, dv_model, good_rtid, bad_rtid, X_embedded_3, colors_dict, pair_labels_dict, save_fig=False):
    # note: ONLY works for texts in training corpus, as of now... come back and work on this!
    all_info_df = get_pairs_dist_info_df(dv_model, good_rtid, bad_rtid)

    sorted_cs_df = get_sorted_dist_info_df(all_info_df, use_cs=True)
    sorted_ed_df = get_sorted_dist_info_df(all_info_df, use_cs=False)

    # Can maybe pull the right labels / colors via more extensive artist_name, color, idx dicts

    fig = plt.figure(figsize=(30,10))

    n_docs = X_embedded_3.shape[0]
    n_pairs = n_docs / 2
    bar_width = 0.5
    opacity = 0.5
    index = np.arange(n_pairs)

    ax1 = fig.add_subplot(131)
    rects1 = ax1.bar(index, sorted_ed_df['pair_ed'], bar_width,
                    alpha=opacity)
    # ALSO NEED TO FIX THIS PART PLS
    ed_pair_labels = ['Hamilton', 'Eminem-Court', 'Eminem-OCD', 'Kanye', 'JCole', 'Kendrick']
    ed_colors = ['red', 'orange', 'yellow', 'purple', 'blue', 'green']
    for idx in range(n_pairs):
        rects1[idx].set_color(ed_colors[idx])
        rects1[idx].set_label(ed_pair_labels[idx])
    ax1.set_xlabel('Pairs', fontsize=16)
    ax1.set_ylabel('Euclidean Distance', fontsize=16)
    ax1.set_title('Euclidean Distances by Pair', fontsize=20)
    ax1.set_xticks(index + bar_width / 2)
    ax1.set_xticklabels(ed_pair_labels)

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title("Doc2Vec 3D Representations of Annotations & Lyrics", fontsize=22)
    X = X_embedded_3[:,0]
    y = X_embedded_3[:,1]
    z = X_embedded_3[:,2]
    size = 200
    n_docs = X_embedded_3.shape[0]
    n_pairs = n_docs / 2
    for idx in range(n_docs):
        is_odd_idx = idx % 2 != 0
        alpha = 0.6 if is_odd_idx else 0.3
        ax2.scatter(X[idx], y[idx], z[idx], s=size, alpha=alpha, c=colors_dict[idx], label=pair_labels_dict[idx])
        if is_odd_idx:
            label_text = "Distance between {0} pair".format(pair_labels_dict[idx])
            line_format = ':' + colors_dict[idx][:1]
            plt.plot([X[idx-1], X[idx]], [y[idx-1], y[idx]], [z[idx-1], z[idx]], line_format, label=label_text);
    ax2.legend(loc='upper center', ncol=int(n_pairs), markerscale=0.5, bbox_to_anchor=(0.5, -0.15))

    ax3 = fig.add_subplot(133)
    rects3 = ax3.bar(index, sorted_cs_df['pair_cs'], bar_width,
                    alpha=opacity)
    # ALSO NEED TO FIX THIS PART PLS
    cs_pair_labels = ['Eminem-OCD', 'Kendrick', 'JCole', 'Kanye', 'Eminem-Court', 'Hamilton']
    cs_colors = ['yellow', 'green', 'blue', 'purple', 'orange', 'red']
    for idx in range(n_pairs):
        rects3[idx].set_color(cs_colors[idx])
        rects3[idx].set_label(cs_pair_labels[idx])
    ax3.set_xlabel('Pairs', fontsize=16)
    ax3.set_ylabel('Cosine Similarity', fontsize=16)
    ax3.set_title('Cosine Similarities by Pair', fontsize=20)
    ax3.set_xticks(index + bar_width / 2)
    ax3.set_xticklabels(cs_pair_labels)

    plt.show()

    if save_fig:
        file_name = '../images/{}_3d_tsne_dist_comp_plot.png'.format(groupname)
        fig.savefig(file_name)


if __name__ == '__main__':
    # Referent and Annotation DocVecs stored in dictionary - keys = rt_ids, values = docvec
    corpus_dict = read_in_pkl('corpus_dict')
    train_df = read_in_pkl('train_df')
    test_df = read_in_pkl('test_df')
    lookup_dicts = read_in_pkl('lookup_dicts')
    pcorpuses = read_in_pkl('pcorpuses')

    train_pairings_df = read_in_pkl('train_pairings_df')
    test_pairings_df = read_in_pkl('test_pairings_df')

    # eval_df = read_in_pkl('eval_df')
    eval_df = pd.read_csv('../data/eval_df_i.csv')

    artists = lookup_dicts[0]
    doc_rt_train_dict = lookup_dicts[1]
    doc_rt_test_dict = lookup_dicts[2]
    rt_doc_train_dict = lookup_dicts[3]
    rt_doc_test_dict = lookup_dicts[4]
    rt_in_train_dict = lookup_dicts[5]
    artist_rt_dict = lookup_dicts[6]
    rt_artist_dict = lookup_dicts[7]
    song_rt_dict = lookup_dicts[8]
    rt_song_dict = lookup_dicts[9]
    song_id_to_title_dict = lookup_dicts[10]

    ref_train_pcorpus = pcorpuses[0]
    ref_test_pcorpus = pcorpuses[1]
    tate_train_pcorpus = pcorpuses[2]
    tate_test_pcorpus = pcorpuses[3]
    rt_train_pcorpus = pcorpuses[4]
    rt_test_pcorpus = pcorpuses[5]
    rt_tagged_train_pcorpus = pcorpuses[6]
    rt_tagged_test_pcorpus = pcorpuses[7]

    rt_50 = Doc2VecModeler('rt_50', rt_train_pcorpus, 'rt', vector_size=50)
    rt_50.fit_model(verbose=True)

    rt_50.calc_self_recognition_ability()

    rt_50.calculate_pair_sims_array(train_pairings_df, is_train=True)
    rt_50.calculate_pair_sims_array(test_pairings_df, is_train=False)

    rt_50.calc_tt_hypothesis_test(for_train_pairings=True, for_cs=True)
    rt_50.calc_tt_hypothesis_test(for_train_pairings=False, for_cs=True)
    rt_50.calc_tt_hypothesis_test(for_train_pairings=True, for_cs=False)
    rt_50.calc_tt_hypothesis_test(for_train_pairings=False, for_cs=False)

    rt_50.plot_sim_distribution(use_train=True, use_cs=True, save_fig=True)
    rt_50.plot_sim_distribution(use_train=False, use_cs=True, save_fig=True)
    rt_50.plot_sim_distribution(use_train=True, use_cs=False, save_fig=True)
    rt_50.plot_sim_distribution(use_train=False, use_cs=False, save_fig=True)

    # HAND PICK GOOD/BAD EXAMPLES -- this stuff is hardcoded
    good_rtid = [5057218, 14369122, 14695297]
    bad_rtid = [9176821, 13660449, 9224654]

    good_doc_ids = get_doc_ids(good_rtid, rt_doc_train_dict)
    bad_doc_ids = get_doc_ids(bad_rtid, rt_doc_train_dict)

    good_pp_ref_txts = get_pp_texts(good_doc_ids, ref_train_pcorpus)
    good_pp_tate_txts = get_pp_texts(good_doc_ids, tate_train_pcorpus)
    bad_pp_ref_txts = get_pp_texts(bad_doc_ids, ref_train_pcorpus)
    bad_pp_tate_txts = get_pp_texts(bad_doc_ids, tate_train_pcorpus)

    good_ref_vecs = get_inferred_vecs(rt_50, good_pp_ref_txts)
    good_tate_vecs = get_inferred_vecs(rt_50, good_pp_tate_txts)
    bad_ref_vecs = get_inferred_vecs(rt_50, bad_pp_ref_txts)
    bad_tate_vecs = get_inferred_vecs(rt_50, bad_pp_tate_txts)

    all_vecs_arr = combine_vecs(good_ref_vecs, good_tate_vecs, bad_ref_vecs, bad_tate_vecs)

    X_embedded_2 = get_tsne_representations(all_vecs_arr, 2)
    X_embedded_3 = get_tsne_representations(all_vecs_arr, 3)

    # Gotta find a way to do this w/o hardcoding...
    colors_dict = {0: 'green', 1: 'green',
                2: 'blue', 3:'blue',
                4: 'purple', 5: 'purple',
                6: 'yellow', 7: 'yellow',
                8: 'red', 9: 'red',
                10: 'orange', 11: 'orange'}

    pair_labels_dict = {0: 'Kendrick', 1: 'Kendrick',
                2: 'JCole', 3: 'JCole',
                4: 'Kanye', 5: 'Kanye',
                6: 'Eminem-OCD', 7: 'Eminem-OCD',
                8: 'Hamilton', 9: 'Hamilton',
                10: 'Eminem-Court', 11: 'Eminem-Court'}

    group_name = 'initial_handpicked'

    plot_2d_version(group_name, X_embedded_2, colors_dict, pair_labels_dict, save_fig=False)

    plot_3d_version(group_name, X_embedded_3, colors_dict, pair_labels_dict, save_fig=False)

    plot_cs_ed_3d_comparison(group_name, rt_50, good_rtid, bad_rtid, X_embedded_3, colors_dict, pair_labels_dict, save_fig=False)
