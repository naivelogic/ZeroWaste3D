"""
WIP
"""

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on：", device)


    net.to(device)
    torch.backends.cudnn.benchmark = True

    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []

    for epoch in range(num_epochs+1):

        t_epoch_start = time.time()
        t_iter_start = time.time()

        print('-------------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train() 
                print('（train）')
            else:
                if((epoch+1) % 10 == 0):
                    net.eval()
                    print('-------------')
                    print('（val）')
                else:
                    
                    continue

            for images, targets in dataloaders_dict[phase]:


                images = images.to(device)
                targets = [ann.to(device) for ann in targets]

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(images)

                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c


                    if phase == 'train':
                        loss.backward()

                        
                        nn.utils.clip_grad_value_(
                            net.parameters(), clip_value=2.0)

                        optimizer.step()

                        if (iteration % 10 == 0):  
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print('>> training iter {} || Loss: {:.4f} || 10iter: {:.4f} sec.'.format(
                                iteration, loss.item(), duration))
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item()
                        iteration += 1
                        
                    else:
                        epoch_val_loss += loss.item()

        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(
            epoch+1, epoch_train_loss, epoch_val_loss))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        log_epoch = {'epoch': epoch+1,
                     'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("../outputs/log_output_dev3_060120.csv")

        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        
        if ((epoch+1) % 10 == 0):
            torch.save(net.state_dict(), ML_WEIGHTS_PATH + 'dev3_ssd300_' +
                       str(epoch+1) + '_060120.pth')