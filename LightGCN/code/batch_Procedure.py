def train_joint(model, dataset, old_embeddings, opt_lgcn, old_knowledge, epoch, ww=None):
    Meta_model=model
    Meta_model.train()

    S, sam_time = utils.UniformSample_handle( dataset, world.sample_mode)
    print(f"BPR[lgcn handle sample time][{sam_time:.2f}]")
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()
    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    aver_loss1 = 0.
    aver_loss2 = 0.
    aver_icl_regloss = 0.

    (old_User, old_Item) = old_knowledge
    del old_User

    for (batch_i,(batch_users, batch_pos, batch_neg)) in enumerate(utils.minibatch(users, posItems, negItems, batch_size=world.config['bpr_batch_size'])):

        knn_0 = old_Item[batch_pos.long()] @ old_Item.t() 
        knn_1 = old_Item[batch_pos.long()].pow(2).sum(dim=1).unsqueeze(1)
        knn_2 = old_Item.pow(2).sum(dim=1).unsqueeze(0)
        knn = knn_1 + knn_2 -2 * knn_0
        knn = knn.pow(2)
        knn[:,dataset.active_item_now]=np.inf
        batch_mtach_items_itself = torch.topk(knn, world.icl_k+1, largest=False, dim=1)[1]
        batch_mtach_items = batch_mtach_items_itself[:,1:world.icl_k+1]

        del knn_0, knn_1, knn_2, knn 

        loss, loss1, loss2, icl_regloss = Meta_model.get_our_loss(old_embeddings, batch_users, batch_pos, batch_neg, batch_mtach_items)
        opt_lgcn.zero_grad()
        loss.backward()
        opt_lgcn.step()
        aver_loss+=loss.cpu().item()
        aver_loss1+=loss1.cpu().item()
        aver_loss2+=loss2.cpu().item()
        aver_icl_regloss+=icl_regloss.cpu().item()

    aver_loss = aver_loss / total_batch
    aver_loss1 = aver_loss1 / total_batch
    aver_loss2 = aver_loss2 / total_batch
    aver_icl_regloss = aver_icl_regloss / total_batch
    return f"[Train aver loss{aver_loss:.4e} = train_loss {aver_loss1:.4e} + icl_loss  {aver_loss2:.4e} + icl_regloss  {aver_icl_regloss:.4e} + reg]"