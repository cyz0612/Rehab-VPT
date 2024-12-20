import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import plotly.graph_objects as go
import plotly.io as pio
import kaleido
import plotly.express as px

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true,  week_num,preds=None, fig_name='./pic/test.pdf'):
    """
    Results visualization
    """
    # plt.figure()
    # plt.plot(true,'-o',label='Real plan', linewidth=1,markersize=3)
    # if preds is not None:
    #     plt.plot(preds,'-o',label='Predicted load', linewidth=1,markersize=3)
    # plt.legend()
    # plt.xlabel("days")
    # plt.ylabel("load (kg)")
    # plt.title("Iteration "+ str(week_num))
    # plt.savefig(fig_name, bbox_inches='tight',dpi=300)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Real plan",
            x=np.array(range(1,len(true)+1)),
            y=true,
            mode="markers+lines",
            marker_size=5
    ))
    fig.add_trace(
        go.Scatter(
            name="Predicted load",
            x=np.array(range(1,len(preds)+1)),
            y=preds,
            mode="markers+lines",
            marker_size=5  
    ))
    
    fig.update_layout(
        title="Iteration "+ str(week_num),
        xaxis=dict(title="days", nticks=13),
        yaxis=dict(title="load (kg)", nticks=11, rangemode="tozero"),
        width=800,
        height=500
    )
    pio.write_image(fig,fig_name,scale=5, width=800, height=500)


def visual(true,  week_num,preds=None, fig_name='./pic/test.pdf'):
    """
    Results visualization
    """
    # plt.figure()
    # plt.plot(true,'-o',label='Real plan', linewidth=1,markersize=3)
    # if preds is not None:
    #     plt.plot(preds,'-o',label='Predicted load', linewidth=1,markersize=3)
    # plt.legend()
    # plt.xlabel("days")
    # plt.ylabel("load (kg)")
    # plt.title("Iteration "+ str(week_num))
    # plt.savefig(fig_name, bbox_inches='tight',dpi=300)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Real plan",
            x=np.array(range(1,len(true)+1)),
            y=true,
            mode="markers+lines",
            marker_size=5
    ))
    fig.add_trace(
        go.Scatter(
            name="Predicted load",
            x=np.array(range(1,len(preds)+1)),
            y=preds,
            mode="markers+lines",
            marker_size=5  
    ))
    
    fig.update_layout(
        title="Iteration "+ str(week_num),
        xaxis=dict(title="days", nticks=13),
        yaxis=dict(title="load (kg)", nticks=11, rangemode="tozero"),
        width=800,
        height=500
    )
    pio.write_image(fig,fig_name,scale=5, width=800, height=500)

def visual2(true,week_num,v1,v2,v3,v4,v5,preds=None,fig_name='./pic/test.pdf'):
    """
    Results visualization
    """

    fig = go.Figure()
    csq = px.colors.qualitative.G10
    fig.add_trace(
        go.Scatter(
            name="Suggested load",
            x=np.array(range(1,len(true)+1)),
            y=true,
            mode="markers+lines",
            marker_size=8,
            line =dict(color = "#3366CC")
    ))
    fig.add_trace(
        go.Scatter(
            name="Predicted load",
            x=np.array(range(1,len(preds)+1)),
            y=preds,
            mode="markers+lines",
            marker_size=8,
            line = dict(color = "#DC3912")
    ))
    fig.add_trace(
        go.Scatter(
            name="Pain level",
            x=np.array(range(1,len(v1)+1)),
            y=v1,
            mode="markers+lines",
            marker_size=8,
            line = dict(color = "#FF9900"),
        yaxis="y2"  # Assigning this trace to the right-side y-axis
    ))
    fig.add_trace(
        go.Scatter(
            name="Success rate",
            x=np.array(range(1,len(v3)+1)),
            y=v3,
            mode="markers+lines",
            marker_size=8,
            # line = dict(color = "#990099"),
            line = dict(color = "#8C564B"),
        yaxis="y2"  # Assigning this trace to the right-side y-axis
    ))
    fig.add_trace(
        go.Scatter(
            name="Adverse reaction",
            x=np.array(range(1,len(v2)+1)),
            y=v2,
            mode="markers+lines",
            marker_size=8,
            line = dict(color = "#109618"),
        yaxis="y2"  # Assigning this trace to the right-side y-axis
    ))
    fig.add_trace(
        go.Scatter(
            name="Training score",
            x=np.array(range(1,len(v4)+1)),
            y=v4,
            mode="markers+lines",
            marker_size=8,
            line = dict(color = "#0099C6"),
        yaxis="y2"  # Assigning this trace to the right-side y-axis
    ))
    fig.add_trace(
        go.Scatter(
            name="Actual load",
            x=np.array(range(1,len(v5)+1)),
            y=v5,
            mode="markers+lines",
            marker_size=8,
            # line = dict(color = "#66AA00"),
            # line = dict(color = "#990099"),
            line = dict(color = "#9467BD"),
    ))

    
    fig.update_layout(
        title="Iteration "+ str(week_num),
        # template="ggplot2",
        template="simple_white",
        xaxis=dict(title="<b>days</b>", nticks=13),
        yaxis=dict(title="<b>load (kg)</b>", nticks=11, rangemode="tozero"),
        yaxis2=dict(title="<b>evaluation values</b>",range=[-1, 28],overlaying='y',side='right',title_standoff=0),
        width=800,
        height=500,
        legend=dict(
        x=1.08,  # Adjust the x position of the legend
        y=1  # Adjust the y position of the legend
        ),
        showlegend=False
    )
    # if len(true)+1 ==23:
    pio.write_image(fig,fig_name,scale=5, width=700, height=500)

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))