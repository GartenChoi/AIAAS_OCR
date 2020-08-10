# doc, hwp, pdf, gif, png, jpg, ...
# 스캔된 이미지 -> 분류 -> 회전 -> 중요한 부분 탐지 -> ocr

from base import *
import os
from mpad.utils import load_file, load_embeddings, get_vocab, create_gows, generate_batches, normalize, parse_doc
from mpad.models import MPAD

def predict(doc):
    docs, class_labels = load_file()
    enc = LabelEncoder()
    class_labels = enc.fit_transform(class_labels)
    nclass = np.unique(class_labels).size
    y = list()
    for i in range(len(class_labels)):
        t = np.zeros(1)
        t[0] = class_labels[i]
        y.append(t)
    vocab = get_vocab(docs)
    embeddings = load_embeddings(docs)
    adj, features, _ = create_gows(docs, vocab, WINDOW_SIZE, DIRECTED, NORMALIZE, USE_MASTER_NODE)

    adj_test = [adj[0]]
    features_test = [features[0]]
    y_test = [y[0]]

    adj_test, features_test, batch_n_graphs_test, y_test = generate_batches(adj_test, features_test, y_test, BATCH_SIZE,USE_MASTER_NODE)

    # Model and optimizer
    model = MPAD(embeddings.shape[1], MESSAGE_PASSING_LAYERS, HIDDEN_NUM, PENULTIMATE, nclass, DROPOUT, embeddings,
                 USE_MASTER_NODE)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=LR)
    if os.path.isfile(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    if CUDA:
        model.cuda()
        adj_test = adj_test.cuda()
        features_test = features_test.cuda()
        batch_n_graphs_test = batch_n_graphs_test.cuda()
        y_test = y_test.cuda()

    def test(adj, features, batch_n_graphs, y):
        output = model(features, adj, batch_n_graphs)
        loss_test = F.cross_entropy(output, y)
        return output, loss_test

    # Testing
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    output, loss = test(adj_test[0], features_test[0], batch_n_graphs_test[0], y_test[0])
    print(output.data)
    print(y_test[0].data)

if __name__ == '__main__':
    for filename in os.listdir(SAMPLE_DATA_DIR):
        print(filename)
        predict(parse_doc(filename))
