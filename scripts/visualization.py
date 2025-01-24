import plotly.graph_objects as go

def create_interactive_chart(df):
    fig = go.Figure()
    
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        )
    )
    
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        template='plotly_dark',
        height=600
    )
    
    return fig

def add_volume_analysis(fig, df):
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            yaxis='y2'
        )
    )
    
    fig.update_layout(
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right"
        )
    )
    return fig

def plot_forecast(df, forecast_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Historical Price'))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['XGBoost_Prediction'], name='XGBoost Forecast', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Hybrid_Prediction'], name='Hybrid Forecast', line=dict(dash='dot')))
    fig.update_layout(template='plotly_dark', height=400)
    return fig

