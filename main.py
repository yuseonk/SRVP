import argparse
from utils import *
from datasets.moving_mnist import MovingMNIST
from model import SRVP


def parse_args():
    parser = argparse.ArgumentParser(description='experiments')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--vizonly', type=bool, default=False)
    parser.add_argument('--ctx', type=int, default=0)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--layer', type=int, default=4)
    args = parser.parse_args()
    return args


def get_model():
    global input_time, horizon, channels, hidden, num_layers, height, width, batch_size, device
    net = None
    hidden_channels = [hidden for _ in range(num_layers)]
    kernel_size = [5 for _ in range(num_layers)]
    dropout = 0.5
    emd_dim=hidden//2
    net = SRVP.EncoderDecoder(emd_dim, input_time, channels, hidden_channels, kernel_size, (height, width), dropout, horizon, channels, device)
    return net


def viz(batch_idx, input_time, inputs, groundtruth, predictions, dst_path):
    x = inputs[batch_idx,...]
    y = groundtruth[batch_idx,...]
    y_hat = predictions[batch_idx,...]
        
    fig, axes = plt.subplots(figsize=(20, 10), nrows=3, ncols=input_time)
    fontsize = 10
    vmin, vmax = 0, 255
    i = 0
    for i in range(input_time):
        ax = axes[0][i]
        img = ax.imshow(x[i, ...],
                        cmap=cm.gray, vmin=vmin, vmax=vmax, interpolation='none')
        ax.set_title(f'$X_{({i+1})}$', fontsize=fontsize)
        ax.axis('off')
        
        ax = axes[1][i]
        img = ax.imshow(y[i, ...],
                        cmap=cm.gray, vmin=vmin, vmax=vmax, interpolation='none')
        ax.set_title(f'$X_{({input_time+i+1})}$', fontsize=fontsize)
        ax.axis('off')
    
        ax = axes[2][i]
        img = ax.imshow(y_hat[i, ...],
                        cmap=cm.gray, vmin=vmin, vmax=vmax, interpolation='none')
        ax.axis('off')
        
        i+=1
    plt.savefig(os.path.join(dst_path, f'spatiotemp_eval{batch_idx}.png'), dpi=300, transparent=True, bbox_inches='tight')


if __name__ == "__main__":
    args = parse_args()

    # set seed
    ctx = args.ctx
    Tensor = torch.FloatTensor
    cuda = True if torch.cuda.is_available() else False
    device = torch.device(f'cuda:{ctx}' if cuda else 'cpu')
    seed = 42
    set_seed(seed)

    # hyper parameters
    exp = 'mnist'
    batch_size = args.bs
    epoch = args.epoch
    lr = args.lr
    horizon = args.horizon
    if horizon != 10:
        exp = f'mnist_{str(horizon)}'
    hidden = args.hidden
    num_layers = args.layer

    # folder path
    if args.mode == 'train':
        dst_path = f'./results'
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        dst_path = os.path.join(dst_path, exp)
        if os.path.isdir(dst_path):
            shutil.rmtree(dst_path)
            os.mkdir(dst_path)
        else:
            os.mkdir(dst_path)
    else:
        dst_path = f'./results/{exp}/'

    # logger setting
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join(dst_path, 'train.log')
    if args.mode == 'train':
        if os.path.isfile(log_file_path):
            os.remove(log_file_path)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    if args.mode == 'train':
        logger.info(args)

    # data loader
    root='./datasets/'
    win_size = 10
    trainset = MovingMNIST(root=root, is_train=True, n_frames_input=win_size, n_frames_output=horizon, num_objects=[2])
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    testset = MovingMNIST(root=root, is_train=False, n_frames_input=win_size, n_frames_output=horizon, num_objects=[2])
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

    for i, sample in enumerate(train_loader):
        if args.mode == 'train':
            logger.info(f"input shape, {sample['input'].shape}, target shape, {sample['label'].shape}")
        break
    _, input_time, channels, height, width = sample['input'].shape

    if args.mode == 'train':
        # modeling
        net = get_model().to(device)
        logger.info(net)

        # training
        trainer = Estimator(epoch=epoch, bs=batch_size, net=net,
                            train_set=train_loader, val_set=test_loader, ctx=ctx, dst_path=dst_path)
        logger.info('Training')
        trainer.do(logger, lr)

    if args.mode == 'test':        
        # evaluation
        torch.cuda.empty_cache()
        gc.collect()

        if args.vizonly:
            groundtruth = np.load(os.path.join(dst_path, f'groundtruth.npy'))
            predictions = np.load(os.path.join(dst_path, f'predictions.npy'))
        else:
            net = get_model()
            test_eval = Evaluator(net=net, test_set=test_loader, ctx=ctx)
            predictions, groundtruth = test_eval.inference(dst_path)
            np.save(os.path.join(dst_path, f'groundtruth.npy'), groundtruth)
            np.save(os.path.join(dst_path, f'predictions.npy'), predictions)

            logger.info("spatiotemporal accuracy")
            score = pd.DataFrame()
            for h in range(1, horizon + 1):
                score = pd.concat([score, get_score(predictions[:, h - 1, ], groundtruth[:, h - 1, ], h)])
            score.to_csv(os.path.join(dst_path, 'spatiotemp_score.csv'))
            logger.info(score)
        
        inputs = np.concatenate([data['input'].numpy() for data in list(test_loader)])
        inputs = np.transpose(inputs, (0, 1, 3, 4, 2))
        inputs = float_to_uint(inverse_scale(inputs))

        # fig
        batch_idx = 80
        viz(batch_idx, input_time, inputs, groundtruth, predictions, dst_path)