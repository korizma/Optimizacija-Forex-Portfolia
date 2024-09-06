import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as st
import scipy.optimize as sco

def expRet(ret,n=252,com=False):
    """Calculates expected returns in one of two methods. Parameters are
    - ret: Pandas' DataFrame or Series of returns
    - n: optional, number of compounding periods in a year (252 for days, 52 for weeks, 12 for months, 4 for quarters...)
    - com: optional, deterimines whether expected returns are calculated as annualized sample mean or annualized compunded return"""
    if com:
        return (1+ret).prod()**(n/len(ret))-1
    else:
        return ret.mean()*n
    
def annualize_vol(ret, n=252):
    """Calculates volatility of sample returns. Parameters are:
    - ret: Pandas' DataFrame or Series of returns
    - n: optional, number of compounding periods in a year (252 for days, 52 for weeks, 12 for months, 4 for quarters...)"""
    return ret.std()*(n**0.5)

def sharpe(ret,n=252,rf=0,com=False):
    """Calculates Sharpe's ratio. Parameters are:
    - ret: Pandas' DataFrame or Series of returns
    - rf: optional, risk free rate (should be given as decimal, if it is omitted, we assume it to be zero)
    - n: optional, number of compounding periods in a year (252 for days, 52 for weeks, 12 for months, 4 for quarters...)
    - com: optional, deterimines whether expected returns are calculated as annualized sample mean or annualized compunded return"""
    return (expRet(ret,n,com)-rf)/annualize_vol(ret,n)

def hist(df,CDF=False):
    """For the given DataFrame or Series of returns function plot histogram along with appropriate normal curve for each colummn.
    Parameters are:
    - df: DataFrame or Series of returns.
    - CDF: If True is given returns comparison of empirical and theoretical CDF, while if False is given returns comparison of
    theoretical and empirical PDF"""
    if str(type(df))!="<class 'pandas.core.frame.DataFrame'>" and str(type(df))!="<class 'pandas.core.series.Series'>":
        return print("Wrong input! Please insert data as Pandas' Series or DataFrame object")
    elif str(type(df))=="<class 'pandas.core.series.Series'>":
        df=df.to_frame()
    else:
        pass
    n=df.shape[1]
    if n==1:
        Row=1
        fig = make_subplots(rows=1, cols=1) 
    elif n%2==0:
        Row=int(n/2)
        fig = make_subplots(rows=Row, cols=2,subplot_titles=df.columns,shared_yaxes=True,specs=[[{}, {}]]*Row) 
    else:
        Row=int((n-1)/2)+1
        fig = make_subplots(rows=Row, cols=2,subplot_titles=df.columns,
                            shared_yaxes=True,specs=[[{}, {}]]*(Row-1) +[[{"colspan": 2}, None]])
    l_Row=[item for sublist in [[i,i] for i in range(1,Row+1)] for item in sublist]
    
    for i in range(n):
        r=df.iloc[:,i]
        x_list=(np.linspace(min(r),max(r),100) if CDF else np.linspace(r.mean()-3*r.std(),r.mean()+3*r.std(),100))
        y_list=(st.norm.cdf(x_list,r.mean(),r.std()) if CDF else st.norm.pdf(x_list,r.mean(),r.std())/100)
        fig.add_trace(go.Histogram(x=r, marker=dict(color='Orange',line=dict(width=2,color='black')),
            histnorm='probability',cumulative=dict(enabled=CDF)),row=l_Row[i],col=(2 if (i+1)%2==0 else 1))
        fig.add_trace(go.Scatter(x=x_list,y=y_list,line_color='DarkGreen',fill='tozeroy'),
                      row=l_Row[i],col=(2 if (i+1)%2==0 else 1))
        fig.update_xaxes(zerolinecolor='black',row=l_Row[i],col=(2 if (i+1)%2==0 else 1))
        fig.update_yaxes(zerolinecolor='black',row=l_Row[i],col=(2 if (i+1)%2==0 else 1))
    fig.update_layout(title=dict(text='Histograms and normal curve',font=dict(size=30),x=0.5,y=0.95),showlegend=False)

    return fig.show()

def semidev(df,n=252,zeromean=False):
    """For the given DataFrame of returns function calculates annualized semideviation for each column. Paramters are:
    - df: DataFrame or Series of returns.
    - n: optional, annualization factor (252 for daily data, 12 for monthly...). If you don't font annualization, set it to 1.
    - zeromean: optional. If True: assumes that mean is zero. If False: function calculate mean from the data."""
    if str(type(df))!="<class 'pandas.core.frame.DataFrame'>" and str(type(df))!="<class 'pandas.core.series.Series'>":
        return print("Wrong input! Please insert data as Pandas' Series or DataFrame object")
    elif str(type(df))=="<class 'pandas.core.series.Series'>":
        df=df.to_frame()
    else:
        pass
    dfN=pd.DataFrame()
    for i in range(df.shape[1]):
        r=df.iloc[:,i]
        m=(0 if zeromean else r.mean())
        dfN[df.columns[i]]=[np.sqrt(np.mean((r[r<m]-m)**2))*(n**0.5)]
    dfN.index=['Semideviation']
    return dfN

def sortino(ret,n=252,rf=0,com=False):
    """Calculates Soritano's ratio. Parameters are:
    - ret: Pandas' DataFrame or Series of returns
    - rf: optional, risk free rate (should be given as decimal, if it is omitted, we assume it to be zero)
    - n: optional, number of compounding periods (annualization factor) in a year (252 for days, 52 for weeks, 12 for months...)
    - com: optional, deterimines whether expected returns are calculated as annualized sample mean or annualized compunded return"""
    return (expRet(ret,n,com)-rf)/semidev(ret,n).rename(index={'Semideviation':'Sortino'})

def per(W,ret=None,n=252,com=False,er_assumed=None):
    """Calculates expected returns of portfoli:
    - ret: data set of returns
    - W: iterable of weights
    - n: optional, number of compounding periods in a year (252 for days, 52 for weeks, 12 for months, 4 for quarters...)
    - com: optional, deterimines whether expected returns are calculated as annualized sample mean or annualized compunded retu
    - er_assumed: optional, assummed expected returns on each stock"""
    er=expRet(ret,n,com) if np.all(er_assumed==None) else er_assumed
    return er@W

def pV(W,ret=None,n=252,vol=False,cov_assumed=None):
    """Calculates variance returns of portfoli:
    - ret: data set of returns
    - W: iterable of weights
    - n: optional, number of compounding periods in a year (252 for days, 52 for weeks, 12 for months, 4 for quarters...)
    - vol: optional, determines whether you want to display portfolio's volatility (True) or variance (False)
    - cov_assumed: optional, assummed annualized covariance matrix of returns"""
    CovM=(ret.cov()*n if np.all(cov_assumed==None) else cov_assumed)
    return np.sqrt(W@CovM@W) if vol else W@CovM@W

def mvp(ret=None,f=252,com=False,cov_assumed=None):
    """Calculates MVP portfolio weights, volatility and expected returns for given data set of returns. Paramters are:
    - ret: optional, data set of returns. If it isn't provided, than it is required to provide er_assumed and cov_assumed.
    - f: optional, number of compounding periods in a year, i.e. frequency (252 for days, 52 for weeks, 12 for months, 4 for quarters...)
    - com: optional, deterimines whether expected returns are calculated as annualized sample mean or annualized compunded ret
    - cov_assumed: optional, assummed annualized covariance matrix of returns"""
    n=(len(ret.columns) if np.all(ret!=None) else len(cov_assumed))
    result=sco.minimize(lambda w: pV(w,ret,f,False,cov_assumed),[1/n]*n,constraints=[dict(type='eq',fun=lambda w:sum(w)-1)])
    return dict(w=result.x,er=per(result.x,ret,f,com),vol=np.sqrt(result.fun)) if np.all(ret!=None) else result.x

def targetP(ret=None,mi=None,Bounds=None,f=252,com=False,er_assumed=None, cov_assumed=None):
    """Calculates optimal portfolio weights, volatility and expected returns. Paramters are:
    - ret: optional, data set of returns. If it isn't provided, than it is required to provide er_assumed and cov_assumed.
    - mi: target level of expected returns. If mi is omitted, function will caulculate MVP weights!
    - f: optional, number of compounding periods in a year, i.e. frequency (252 for days, 52 for weeks, 12 for months, 4 for quarters...)
    - com: optional, deterimines whether expected returns are calculated as annualized sample mean or annualized compunded retu
    - er_assumed: optional, assummed expected returns on each stock
    - cov_assumed: optional, assummed annualized covariance matrix of returns
    - Bounds: optional, simple list of length 2 (so that elements represent upper and lower bound) or complex list of length n 
    (so that each sublist represent collection of upper and lower bound for each of n stocks separately) """
    n=(len(ret.columns) if np.all(ret!=None) else len(cov_assumed))
    if mi==None:
        result=sco.minimize(lambda w: pV(w,ret,f,False,cov_assumed),[1/n]*n,constraints=[dict(type='eq',fun=lambda w:sum(w)-1)])
        if np.all(ret!=None):
            return dict(w=result.x, er=per(result.x,ret,f,com,er_assumed),vol=np.sqrt(result.fun))
        else:
            return dict(w=result.x,er=(None if np.all(er_assumed==None) else (result.x)@er_assumed),vol=np.sqrt(result.fun))
    elif type(mi)==float or type(mi)==np.float64:
        result=sco.minimize(lambda w: pV(w,ret,f,False,cov_assumed),[1/n]*n,
                            bounds=(None if Bounds==None else [Bounds]*n if len(Bounds)==2 else Bounds),
                        constraints=[dict(type='eq',fun=lambda w:sum(w)-1),
                                     dict(type='eq',fun=lambda w:per(w,ret,f,com,er_assumed)-mi)])
        return dict(w=result.x, er=mi,vol=np.sqrt(result.fun))
    else:
        return print('\33[91mWrong input given for parameter mi!\33[0m')
    
def EF(ret=None,Range=[0.01,0.4],plot=None,f=252,com=False,er_assumed=None,cov_assumed=None):
    """Prepares data for plotting efficient frontier or plot this curve (depending on what user chooses). Paramters are:
    - ret: optional, data set of returns. If it isn't provided, than it is required to provide er_assumed and cov_assumed.
    - Range: optional, range in which you want to see efficient frontier.
    - f: optional, number of compounding periods in a year, i.e. frequency (252 for days, 52 for weeks, 12 for months, 4 for quarters...)
    - com: optional, deterimines whether expected returns are calculated as annualized sample mean or annualized compunded retu
    - plot: optional, determines whether you want to plot efficient frontier or not. Possible values are:
        a) None - on values will be displayed
        b) 'curve' - only curve will be plotted
        c) 'mvp' - curve with MVP will be plotted
        d) 'full' - curve with MVP will be plotted and efficient part will be emphasised
    - er_assumed: optional, assummed expected returns on each stock
    - cov_assumed: optional, assummed annualized covariance matrix of returns"""
    mi=np.arange(Range[0],Range[1]+0.01,0.01)
    sigma=np.array([targetP(ret,m,f=f,com=com,er_assumed=er_assumed,cov_assumed=cov_assumed)['vol'] for m in mi])
    if plot==None:
        return dict(sigma=sigma,er=mi)
    elif plot=='curve':
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=sigma,y=mi,line=dict(color='Blue',width=3)))
        fig.update_layout(xaxis=dict(title_text='\$\sigma$',range=[-0.01,max(0.25,np.max(sigma)+0.01)],zerolinecolor='Black'),
                  yaxis=dict(title_text='$\mu$',zerolinecolor='Black'),
                  title=dict(text="Efficient frontier",x=0.5,y=0.87,font=dict(size=25,color='Navy')))
        return fig
    elif plot=='mvp':
        M=targetP(ret,m,f=f,com=com,er_assumed=er_assumed,cov_assumed=cov_assumed)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=sigma,y=mi,line=dict(color='Blue',width=3)))
        fig.add_trace(go.Scatter(x=[M['vol']],y=[M['er']],marker=dict(color='Red',line=dict(color='Black',width=2),size=10)))
        fig.update_layout(xaxis=dict(title_text='\$\sigma$',range=[-0.01,max(0.25,np.max(sigma)+0.01)],zerolinecolor='Black'),
                  yaxis=dict(title_text='$\mu$',zerolinecolor='Black'),showlegend=False,
                  title=dict(text="Efficient frontier",x=0.5,y=0.87,font=dict(size=25,color='Navy')))
        return fig
    elif plot=='full':
        M=targetP(ret,f=f,com=com,er_assumed=er_assumed,cov_assumed=cov_assumed)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=sigma,y=mi,line=dict(color='Blue',dash='dash'),fill="tozeroy",fillcolor='Pink'))
        fig.add_trace(go.Scatter(x=sigma[mi>M['er']],y=mi[mi>M['er']],line=dict(color='Blue',width=3)))
        fig.add_trace(go.Scatter(x=[M['vol']],y=[M['er']],marker=dict(color='Red',line=dict(color='Black',width=2),size=10)))
        fig.update_layout(xaxis=dict(title_text='\$\sigma$',range=[-0.01,max(0.25,np.max(sigma)+0.01)],zerolinecolor='Black'),
                  yaxis=dict(title_text='$\mu$',zerolinecolor='Black'),showlegend=False,
                  title=dict(text="Efficient frontier",x=0.5,y=0.87,font=dict(size=25,color='Navy')))

        return fig
    else:
        return print("\33[91mWrong input ginven for parameter plot. Expected values are: None,'curve','mvp' or 'full'\33[0m") 
    
def maxSharpe(ret=None,rf=0.0,Bounds=[-1,1],f=252,com=False,er_assumed=None,cov_assumed=None):
    """Calculates weights for portfolio with maximal Sharpe's ratio, its volatility and expected returns. Paramters are:
    - ret: optional, data set of returns on risky assets. If it isn't provided, than it is required to provide er_assumed and cov_assumed.
    - f: optional, number of compounding periods in a year, i.e. frequency (252 for days, 52 for weeks, 12 for months, 4 for quarters...)
    - com: optional, deterimines whether expected returns are calculated as annualized sample mean or annualized compunded retu
    - rf: data set of returns on risk-free assets or assumed constant 
    - er_assumed: optional, assummed expected returns on each stock
    - cov_assumed: optional, assummed annualized covariance matrix of returns"""
    n=(len(ret.columns) if np.all(ret!=None) else len(cov_assumed))
    r_f=(rf if type(rf)==float or type(rf)==np.float64 else expRet(rf))
    result=sco.minimize(lambda w: -(per(w,ret,f,com,er_assumed)-r_f)/pV(w,ret,f,True,cov_assumed),[1/n]*n,bounds=[Bounds]*n,
                        constraints=[dict(type='eq',fun=lambda w:sum(w)-1)]) 
    return dict(w=result.x, er=per(result.x,ret,f,com,er_assumed),vol=pV(result.x,ret,f,True,cov_assumed),sharpe=-result.fun)

