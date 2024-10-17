import torch
from utils.parser import args
from utils import logger
from utils import init_device, init_model, FakeLR
from dataset import CSIDataLoader
from torch.utils.data import RandomSampler, DataLoader, TensorDataset
from datetime import datetime
import os

def init_dir():
    # initial result directory
    dir_name = 'cr{}-n{}-p{}-ep{}'.format(args.cr, args.store_num, args.period, args.epochs)
    if args.debug:
        save_path = os.path.join(args.root, 'debug', args.method, args.scenarios, dir_name)
    else:
        save_path = os.path.join(args.root, 'release', args.method, args.scenarios, dir_name + '-{0:%m%d-%H%M}'.format(datetime.now()))
    if args.evaluate:
        save_path = os.path.join(args.root, 'evaluate', args.method, args.scenarios, dir_name)
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        os.remove(os.path.join(save_path + '/log.txt')) 

    logger.set_file(os.path.join(save_path + '/log.txt'))
    return save_path

def main():
    save_path = init_dir()
        
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))
    logger.info('scenario: {}, cr: {} '.format(args.scenarios, args.cr))

    # Environment initialization
    device, _ = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)

    # Define model
    model = init_model(args)
    model.to(device)

    # Define scenarios
    scenarios = []
    for s in args.scenarios:
        scenarios.append(s)
        
    n = len(scenarios)
    performance = []
    train_loader_list = []
    test_loader_list = []
    train_tiny_loader_list = []

    # loading data
    for i in range(n):
        train_loader, train_tiny_loader, test_loader = CSIDataLoader(
            root=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            scenario=scenarios[i],
            device=device)()
        train_loader_list.append(train_loader)
        train_tiny_loader_list.append(train_tiny_loader)
        test_loader_list.append(test_loader)
    
    # define optimizer and scheduler
    lr_init = 1e-3 if args.scheduler == 'const' else 2e-3
    optimizer = torch.optim.Adam(model.parameters(), lr_init)
    scheduler = FakeLR(optimizer=optimizer)
    
    
    # define the training strategy
    if args.method == 'Alter': # choose 'Alter' for the Alternating optimization framework
        from methods.alter import Alter as Trainer      
        period = args.period   
        store_data = torch.zeros(args.store_num * 2, 2, 32, 32).to(device)  # store the newest two scenarios for knowledge review
        optimizer_store = torch.optim.Adam(model.parameters(), lr_init / 2)  
        scheduler_store = FakeLR(optimizer=optimizer_store)
    else:
        raise InterruptedError    
        
    # Define the training pipeline
    trainer = Trainer(model=model,
                      device=device,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      save_path=save_path,
                      test_freq=1,
                      print_freq=5
                      )
    
    if args.method == 'Alter':
        trainer.add_optim(optimizer_store, scheduler_store)
    
    performance = torch.zeros([n,n])
    
    ##################################### pretrain #########################################  
    # Offline training
    if args.fresh:
        nmse_list = trainer.loop(t=0,
                    epochs=args.epochs,
                    train_loader=train_loader_list[0],
                    test_loader=[test_loader_list[0]]
                    )
        performance[0,0] = nmse_list[0]    
    else:
        trainer._pretrain(pretrain=os.path.join(args.pretrained, f'best_after_{scenarios[0]}.pth'))                          
        nmse = trainer.test(t=0, test_loader=[test_loader_list[0]])
        performance[0,0] = nmse[0]
    
    ################################### transfer #########################################
        
    for t in range(1, n):
        # first test
        nmse = trainer.test(t=t, test_loader=[test_loader_list[t]])    # pure test
        performance[t-1, t] = nmse[0]
        
        # then finetune
        logger.info(f'Start online updating in scenario {t}\n ')
        nmse_list = trainer.loop(t=t,
                        epochs=args.epochs // 4,    # number of epochs for online updating
                        train_loader = train_tiny_loader_list[t],
                        test_loader = test_loader_list[0:t+1]
                        )
        
        for i in range(t+1):
            performance[t,i] = nmse_list[i]
                 
        logger.info(f'End online updating in scenario {t}\n ')
                 
        # Store online collected dataset and perform knowledge review
        if args.method == 'Alter':
            sampler = RandomSampler(train_tiny_loader_list[t].dataset, num_samples=args.store_num)
            index = torch.tensor([i for i in sampler]).to(device)
            store_data[((t-1)%2)*args.store_num:((t-1)%2+1)*args.store_num, :, :, :] = torch.index_select(train_tiny_loader_list[t].dataset.tensors[0], 0, index)
            
            if (t % period == 0) & (t >= 2):
                logger.info(f'Start knowledge review \n')
                store_loader = DataLoader(TensorDataset(store_data), batch_size=len(store_data) // 4, shuffle=True)
                nmse_list = trainer.loop(t=t,                         
                                         epochs=10, # number of epochs for knowledge review
                                         train_loader=store_loader,
                                         test_loader=test_loader_list[0:t+1],
                                         add_train=True,
                                         )
                logger.info(f'End knowledge review \n')
                
    logger.info(f'{scenarios} \n')
    logger.info(f'The performance of online training is:')
    for nn in range(1, n):
        logger.info(f'updating on scenario {scenarios[nn]}: {performance[nn-1, nn]:.2f} dB --> {performance[nn,nn]:.2f} dB')

if __name__ == "__main__":
    main()
