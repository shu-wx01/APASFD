from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,recall_score,precision_score
import torch
from model import Msg2Mail
import numpy as np
import dgl
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# from args import get_args
from utils import get_current_ts, get_args

def eval_epoch(args, logger, g, dataloader, encoder, decoder, msg2mail, loss_fcn, device, num_samples):

    m_ap, m_auc, m_acc = [[], [], []] if 'LP' in args.tasks else [0, 0, 0]

    labels_all = torch.zeros((num_samples)).long()
    logits_all = torch.zeros((num_samples))
    nodeEmb_all = []
    attn_weight_all = torch.zeros((num_samples, args.n_mail))

    m_loss = []
    m_infer_time = []
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        loss = torch.tensor(0)
        # 首先生成一个颜色渐变数组，用于根据批次大小调整颜色深度
        color_gradient = np.linspace(0, 1, 15)
        for batch_idx, (input_nodes, pos_graph, neg_graph, blocks, frontier, current_ts) in enumerate(dataloader):
            n_sample = pos_graph.num_edges()
            start_idx = batch_idx * n_sample
            end_idx = min(num_samples, start_idx + n_sample)

            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device) if neg_graph is not None else None
            if not args.no_time or not args.no_pos:
                current_ts, pos_ts, num_pos_nodes = get_current_ts(args, pos_graph, neg_graph)
                pos_graph.ndata['ts'] = current_ts
            else:
                current_ts, pos_ts, num_pos_nodes = None, None, None

            _ = dgl.add_reverse_edges(neg_graph) if neg_graph is not None else None

            start = time.time()
            emb, attn_weight = encoder(dgl.add_reverse_edges(pos_graph), _, num_pos_nodes)
            #attn_weight_all[start_idx:end_idx] = attn_weight[:n_sample]

            logits, labels, node_emb = decoder(emb, pos_graph, neg_graph)
            end = time.time() - start
            m_infer_time.append(end)

            loss = loss_fcn(logits, labels)
            m_loss.append(loss.item())
            mail = msg2mail.gen_mail(args, emb, input_nodes, pos_graph, frontier, 'val')
            if not args.no_time:
                g.ndata['last_update'][pos_graph.ndata[dgl.NID][:num_pos_nodes]] = pos_ts.to('cpu')
            g.ndata['feat'][pos_graph.ndata[dgl.NID]] = emb.to('cpu')
            g.ndata['mail'][input_nodes] = mail

            labels = labels.long()
            logits = logits.sigmoid()
            if 'LP' in args.tasks:
                pred = logits > 0.5
                m_auc.append(roc_auc_score(labels_all, logits_all).cpu().item())
                m_acc.append(accuracy_score(labels_all, pred).cpu().item())
            else:
                labels_all[start_idx:end_idx] = labels
                logits_all[start_idx:end_idx] = logits
                nodeEmb_all.extend(node_emb)


            if batch_idx % 50 == 0 and batch_idx != 0:
                nodeEmb_all = nodeEmb_all[0: 400]
                labels_all = labels_all[0: 400]

                pca = PCA(n_components=50)
                tsne = TSNE(n_components=2, random_state=21)
                node_emb_np = [emb.detach().numpy() for emb in nodeEmb_all]
                lgs_all = pca.fit_transform(node_emb_np)
                lgs_all = tsne.fit_transform(lgs_all)

                bIdx = int(batch_idx / 50) - 1
                # 获取当前批次的颜色深度
                color = color_gradient[bIdx]
                # 绘制散点图
                plt.scatter(lgs_all[:, 0], lgs_all[:, 1], color=plt.cm.Blues(color), label=f'Batch {bIdx}',
                            alpha=0.5)
                nodeEmb_all = []
                labels_all = []
            #
            # # 将节点嵌入写入到文件
            # write_vectors_to_txt(node_emb, str(batch_idx) + 'nodesEmbedding.txt')
            # # 将标签写入到文件
            # write_vectors_to_txt(torch.unsqueeze(labels, dim=1), str(batch_idx) + 'labels.txt')
    plt.title('Scatter Plot of Node Embeddings by Batches')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()



    pred_all = logits_all > 0.5
    auc = roc_auc_score(labels_all, logits_all)
    acc = accuracy_score(labels_all, pred_all)

    print('总推理时间', np.sum(m_infer_time))
    logger.info(attn_weight_all.mean(0))
    encoder.train()
    decoder.train()
    return auc, acc, np.mean(m_loss)

def get_TPR_FPR_metrics(fprs, tprs, thresholds):
    FPR_limits=torch.tensor([0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01])
    fpr_l, tpr_l, thres_l=[], [], []
    for limits in FPR_limits:
        idx = torch.where(fprs>limits)[0][0].item()
        fpr_l.append(fprs[idx])
        tpr_l.append(tprs[idx])
        thres_l.append(thresholds[idx])
    return fpr_l, tpr_l, thres_l

def print_tp_fp_thres(task, logger, fpr_l, tpr_l, thres_l):
    for i in range(len(fpr_l)):
        logger.info('Task {}:  -- FPR: {:.4f}, TPR: {:.4f}, Threshold: {:.4f}'.format(task, fpr_l[i].cpu().item(), tpr_l[i].cpu().item(), thres_l[i].cpu().item()))
    logger.info('---------------------------------------------------------------------')

# 将向量列表写入到文本文件
def write_vectors_to_txt(vec_list, file_path):
    # 将列表转换为字符串，每个向量以逗号分隔，整个列表以换行符分隔
    vec_str = '\n'.join([','.join(map(str, vec)) for vec in vec_list])

    # 在末尾添加一个换行符
    vec_str += '\n'

    # 将字符串写入到txt文件
    with open(file_path, 'a') as file:
        file.write(vec_str)

# 从文本文件中读取向量列表
def read_vectors_from_txt(file_path):
    vec_list = []
    with open(file_path, 'r') as file:
        for line in file:
            # 按逗号分割字符串，并将每个元素转换为浮点数
            vec = [float(x) for x in line.strip().split(',')]
            vec_list.append(np.array(vec))  # 将列表转换为numpy数组
    return vec_list
