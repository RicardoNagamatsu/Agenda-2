from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go

#importação da biblioteca para trazer os dados do site
from urllib.request import urlretrieve

df_brasil = pd.read_csv('flask/static/brasil.csv')

def preparar_dados():
    print('Preparando base de dados...')
    # fazendo a importação dos dados atualizados e colocando nos seus respectivos csv's
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    urlretrieve(url, 'static/global_cases.csv')
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    urlretrieve(url, 'static/global_deaths.csv')
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
    urlretrieve(url, 'static/global_recovered.csv')
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv'
    urlretrieve(url, 'static/lookup_table.csv')

    # carregando os csv nos dataframes
    recovered_df = pd.read_csv('./static/global_recovered.csv', sep=',')
    deaths_df = pd.read_csv('./static/global_deaths.csv', sep=',')
    confirmed_df = pd.read_csv('./static/global_cases.csv', sep=',')
    population_df = pd.read_csv('./static/population_by_country_2020 (1).csv', sep=',')
    #lookup_table_df = pd.read_csv('lookup_table.csv', sep=',')

    #
    #   TRATANDO E AGREGANDO DADOS DO DATASET ORIGINAL
    #
    # transpondo todas datas de colunas para linhas
    dates = confirmed_df.columns[4:]
    confirmed_df_long = confirmed_df.melt(
        id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
        value_vars=dates,
        var_name='Date',
        value_name='Confirmed'
    )
    deaths_df_long = deaths_df.melt(
        id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
        value_vars=dates,
        var_name='Date',
        value_name='Deaths'
    )
    recovered_df_long = recovered_df.melt(
        id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
        value_vars=dates,
        var_name='Date',
        value_name='Recovered'
    )

    # removendo os dados recuperados do Canadá devido ao problema de incompatibilidade
    recovered_df_long = recovered_df_long[recovered_df_long['Country/Region']!='Canada']

    # Mesclando confirmed_df_long e deaths_df_long
    full_table = confirmed_df_long.merge(
        right=deaths_df_long,
        how='left',
        on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long']
    )

    # Mesclando full_table e recovered_df_long
    full_table = full_table.merge(
        right=recovered_df_long,
        how='left',
        on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long']
    )
    
    # Convertendo de string para datetime
    full_table['Date'] = pd.to_datetime(full_table['Date'])

    # Preenchendo dados com 0 NaN na coluna 'Recovered'
    full_table['Recovered'] = full_table['Recovered'].fillna(0)
    
    # Filtrando dados de navios e removendo
    ship_rows = full_table['Province/State'].str.contains('Grand Princess') | full_table['Province/State'].str.contains('Diamond Princess') | full_table['Country/Region'].str.contains('Diamond Princess') | full_table['Country/Region'].str.contains('MS Zaandam')
    full_ship = full_table[ship_rows]
    full_table = full_table[~(ship_rows)]

    # Active Case = confirmed - deaths - recovered
    full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']

    #
    # AGREGANDO DADOS DE POPULAÇÃO
    #
    # Removendo colunas que não serão utilizadas
    population_df.drop(['Net Change', 'Land Area (Km²)', 'Migrants (net)', 'Yearly Change', 
         'Fert. Rate', 'World Share' ], axis=1, inplace=True)
    # Renomenado colunas conforme dataset original
    population_df.rename(columns={'Country (or dependency)': 'Country/Region'}, inplace=True)
    full_table = full_table.merge(
        right=population_df,
        how='left',
        on=['Country/Region']
    )
    
    print('Base de dados preparada...')
    return full_table

def group_by_geo(full_table, geo):
    # Agrupando dados por 'Date' e 'Country/Region'    
    full_grouped = full_table.groupby(['Date', geo])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
    
    # Adicionando 'New cases', 'New deaths', e 'New Recovery' para obter informações sobre ocorrencias/dia
    temp = full_grouped.groupby([geo, 'Date', ])['Confirmed', 'Deaths', 'Recovered']
    temp = temp.sum().diff().reset_index()
    mask = temp[geo] != temp[geo].shift(1)
    temp.loc[mask, 'Confirmed'] = np.nan
    temp.loc[mask, 'Deaths'] = np.nan
    temp.loc[mask, 'Recovered'] = np.nan    
    # Renomeando colunas
    temp.columns = [geo, 'Date', 'New cases', 'New deaths', 'New recovered']
    # merging new values
    full_grouped = pd.merge(full_grouped, temp, on=[geo, 'Date'])
    # filling na with 0
    full_grouped = full_grouped.fillna(0)
    # fixing data types
    cols = ['New cases', 'New deaths', 'New recovered']
    full_grouped[cols] = full_grouped[cols].astype('int')
    full_grouped['New cases'] = full_grouped['New cases'].apply(lambda x: 0 if x<0 else x)

    # Adicionando colunas interessantes
    full_grouped['Mortality'] = full_grouped['Deaths'] / full_grouped['Confirmed'] * 100

    return full_grouped

def global_graph(df, plotted_var):
    month_end = df[pd.to_datetime(df["Date"]).dt.is_month_end]
    month_end.loc[:,'Month'] = pd.to_datetime(month_end['Date']).dt.to_period('M').astype(str)
    month_end.loc[month_end['Country/Region'] == 'France', 'Lat'] =  48.8032
    month_end.loc[month_end['Country/Region'] == 'France', 'Long'] =  2.3511
    #df = px.data.gapminder().query("year == 2007")
    fig = px.scatter_geo(month_end,
        lon = 'Long',
        lat = 'Lat',
        #color="continent", # which column to use to set the color of markers
        hover_name='Country/Region', # column added to hover information
        animation_frame='Month',
        size=plotted_var, # size of markers
        projection="natural earth",
        title=plotted_var.title()+" Cases")
        
    fig.update_layout(width=1000, height=500, title_x=0.5)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

app = Flask(__name__)
full_df = preparar_dados()
country_df = group_by_geo(full_df, 'Country/Region')
region_df = group_by_geo(full_df, 'Province/State')



@app.route('/')
def home():
    df_filtrado = country_df.groupby('Date').sum().reset_index()
    df_filtrado['Mortality'] = df_filtrado['Deaths'] / df_filtrado['Confirmed'] * 100
    if (df_filtrado.shape[0] > 0):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_filtrado['Date'], y=df_filtrado['Confirmed'], mode='lines', name='Confirmed'))
        fig.add_trace(go.Scatter(x=df_filtrado['Date'], y=df_filtrado['Recovered'], mode='lines', name='Recovered'))
        fig.add_trace(go.Scatter(x=df_filtrado['Date'], y=df_filtrado['Deaths'], mode='lines', name='Deaths'))

        fig.update_layout(width=1000, height=500, title_x=0.5, title='Dados Globais Covid-19')
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('home.html', df=df_filtrado, plot_json=plot_json)

@app.route('/resumo')
def resumo():
    latest_data = full_df[full_df['Date'] == full_df['Date'].max()]
    latest_data.loc[:, 'Confirmed / 1M Pop'] = latest_data['Confirmed'] / (latest_data['Population (2020)']/1000000)
    latest_data.loc[:, 'Deaths / 1M Pop'] = latest_data['Deaths'] / (latest_data['Population (2020)']/1000000)
    latest_data.loc[:, 'Active / 1M Pop'] = latest_data['Active'] / (latest_data['Population (2020)']/1000000)
    latest_data.loc[:, 'Mortality'] = latest_data['Deaths'] / latest_data['Confirmed'] * 100

    confirmed_json = global_graph(full_df, 'Confirmed')
    
    return render_template('resumo.html', df=latest_data, confirmedJSON=confirmed_json)

@app.route('/pesquisa', methods = ['POST', 'GET'])
def pesquisa():
    if request.method == 'POST':
        agrupamento = request.form['agrupamento']
        periodo = request.form['periodo']
        filtro = request.form['filtro']
        dt_inicial = request.form['inicio']
        dt_final = request.form['final']
    else:
        # recebendo digitado no input do html
        agrupamento = request.args.get('agrupamento')
        periodo = request.args.get('periodo')
        filtro = request.args.get('filtro')
        dt_inicial = request.args.get('inicio')
        dt_inicial = dt.datetime.strptime(dt_inicial, '%Y-%m-%d')
        dt_final = request.args.get('final')
        dt_final = dt.datetime.strptime(dt_final, '%Y-%m-%d')
        
    print(f'agrupamento : {agrupamento}')
    print(f'período : {periodo}')
    print(f'pais/região : {filtro}')
    print(f'inicio : {dt_inicial}')
    print(f'final : {dt_final}')

    if agrupamento == 'Country/Region':
        if filtro != '':
            if dt_inicial != '':
                if dt_final != '':
                    print('agrupamento country -> filtro -> dt_inicial -> dt_final')
                    df_filtrado = country_df[(country_df['Country/Region'] == filtro) &
                        (country_df['Date'] >= dt_inicial) & (country_df['Date'] <= dt_final)]
                else:
                    print('agrupamento country -> filtro -> dt_inicial')
                    df_filtrado = country_df[(country_df['Country/Region'] == filtro) &
                        (country_df['Date'] >= dt_inicial)]
            else:
                if dt_final != '':
                    print('agrupamento country -> filtro -> dt_final')
                    df_filtrado = country_df[(country_df['Country/Region'] == filtro) &
                        (country_df['Date'] <= dt_final)]
                else:
                    print('agrupamento country -> filtro')
                    df_filtrado = country_df[(country_df['Country/Region'] == filtro)]
        else:
            if dt_inicial != '':
                if dt_final != '':
                    print('agrupamento country -> dt_inicial -> dt_final')
                    df_filtrado = country_df[(country_df['Date'] >= dt_inicial) & (country_df['Date'] <= dt_final)]
                else:
                    print('agrupamento country -> dt_inicial')
                    df_filtrado = country_df[country_df['Date'] >= dt_inicial]
            else:
                if dt_final != '':
                    print('agrupamento country -> dt_final')
                    df_filtrado = country_df[country_df['Date'] <= dt_final]
                else:
                    print('agrupamento country')
                    df_filtrado = country_df
    elif agrupamento == 'Province/State':        
        if filtro != '':
            if dt_inicial != '':
                if dt_final != '':
                    print('agrupamento province -> filtro -> dt_inicial -> dt_final')
                    df_filtrado = region_df[(region_df['Province/State'] == filtro) &
                        (region_df['Date'] >= dt_inicial) & (region_df['Date'] <= dt_final)]
                else:
                    print('agrupamento province -> filtro -> dt_inicial')
                    df_filtrado = region_df[(region_df['Province/State'] == filtro) &
                        (region_df['Date'] >= dt_inicial)]
            else:
                if dt_final != '':
                    print('agrupamento province -> filtro -> dt_final')
                    df_filtrado = region_df[(region_df['Province/State'] == filtro) &
                        (region_df['Date'] <= dt_final)]
                else:
                    print('agrupamento province -> filtro')
                    df_filtrado = region_df[(region_df['Province/State'] == filtro)]
        else:
            if dt_inicial != '':
                if dt_final != '':
                    print('agrupamento province -> dt_inicial -> dt_final')
                    df_filtrado = region_df[(region_df['Date'] >= dt_inicial) & (region_df['Date'] <= dt_final)]
                else:
                    print('agrupamento province -> dt_inicial')
                    df_filtrado = region_df[region_df['Date'] >= dt_inicial]
            else:
                if dt_final != '':
                    print('agrupamento province -> dt_final')
                    df_filtrado = region_df[region_df['Date'] <= dt_final]
                else:
                    print('agrupamento country')
                    df_filtrado = region_df
    else:
        if filtro in full_df['Country/Region'].unique():
            print('país')
            if dt_inicial != '':
                if dt_final != '':
                    print('filtro país -> dt_inicial -> dt_final')
                    df_filtrado = full_df[(full_df['Country/Region'] == filtro) &
                        (full_df['Date'] >= dt_inicial) & (full_df['Date'] <= dt_final)]
                else:
                    print('filtro país -> dt_inicial')
                    df_filtrado = full_df[(full_df['Country/Region'] == filtro) &
                        (full_df['Date'] >= dt_inicial)]
            else:
                if dt_final != '':
                    print('filtro país -> dt_final')
                    df_filtrado = full_df[(full_df['Country/Region'] == filtro) &
                        (region_df['Date'] <= dt_final)]
                else:
                    print('filtro país')
                    df_filtrado = full_df[full_df['Country/Region'] == filtro]
        elif filtro in full_df['Province/State'].unique():
            print('provincia')
            if dt_inicial != '':
                if dt_final != '':
                    print('filtro province -> dt_inicial -> dt_final')
                    df_filtrado = full_df[(full_df['Province/State'] == filtro) &
                        (full_df['Date'] >= dt_inicial) & (full_df['Date'] <= dt_final)]
                else:
                    print('filtro province -> dt_inicial')
                    df_filtrado = full_df[(full_df['Province/State'] == filtro) &
                        (full_df['Date'] >= dt_inicial)]
            else:
                if dt_final != '':
                    print('filtro province -> dt_final')
                    df_filtrado = full_df[(full_df['Province/State'] == filtro) &
                        (region_df['Date'] <= dt_final)]
                else:
                    print('filtro province')
                    df_filtrado = full_df[full_df['Province/State'] == filtro]
        else:
            if dt_inicial != '':
                if dt_final != '':
                    print('filtro dt_inicial -> dt_final')
                    df_filtrado = full_df[(full_df['Date'] >= dt_inicial) & (full_df['Date'] <= dt_final)]
                else:
                    print('filtro dt_inicial')
                    df_filtrado = full_df[full_df['Date'] >= dt_inicial]
            else:
                if dt_final != '':
                    print('filtro dt_final')
                    df_filtrado = full_df[region_df['Date'] <= dt_final]
                else:
                    print('sem filtro')
                    df_filtrado = full_df
    if periodo == 'semana':
        print('semana')
        weekday = df_filtrado['Date'].min().weekday()
        df_filtrado = df_filtrado[df_filtrado['Date'].dt.dayofweek == weekday]
    elif periodo == 'mês':
        print('mes')
        df_filtrado = df_filtrado[df_filtrado['Date'].dt.is_month_end]

    print(df_filtrado)

    if (df_filtrado.shape[0] > 0):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_filtrado['Date'], y=df_filtrado['Confirmed'], mode='lines', name='Confirmed'))
        fig.add_trace(go.Scatter(x=df_filtrado['Date'], y=df_filtrado['Recovered'], mode='lines', name='Recovered'))
        fig.add_trace(go.Scatter(x=df_filtrado['Date'], y=df_filtrado['Deaths'], mode='lines', name='Deaths'))

        fig.update_layout(width=1000, height=500, title=f'Dados Covid-19: {filtro}', title_x=0.75)
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('home.html', df=df_filtrado, plot_json=plot_json)


@app.route('/brasil')
def brasil():

    return render_template('brasil.html', df=df_brasil)

@app.route('/amapa')
def amapa():

    return render_template('amapa.html', df=df_brasil)

@app.route('/roraima')
def roraima():

    return render_template('roraima.html', df=df_brasil)

@app.route('/AC')
def AC():

    return render_template('AC.html', df=df_brasil)

@app.route('/TO')
def TO():

    return render_template('TO.html', df=df_brasil)

@app.route('/PA')
def PA():

    return render_template('PA.html', df=df_brasil)

@app.route('/RO')
def RO():

    return render_template('RO.html', df=df_brasil)

@app.route('/MA')
def MA():

    return render_template('MA.html', df=df_brasil)

@app.route('/PI')
def PI():

    return render_template('PI.html', df=df_brasil)

@app.route('/CE')
def CE():

    return render_template('CE.html', df=df_brasil)

@app.route('/RN')
def RN():

    return render_template('RN.html', df=df_brasil)

@app.route('/PB')
def PB():

    return render_template('PB.html', df=df_brasil)

@app.route('/PE')
def PE():

    return render_template('PE.html', df=df_brasil)

@app.route('/AL')
def AL():

    return render_template('AL.html', df=df_brasil)

@app.route('/SE')
def SE():

    return render_template('SE.html', df=df_brasil)

@app.route('/BA')
def BA():

    return render_template('BA.html', df=df_brasil)

@app.route('/MT')
def MT():

    return render_template('MT.html', df=df_brasil)

@app.route('/AM')
def AM():

    return render_template('AM.html', df=df_brasil)

@app.route('/DF')
def DF():

    return render_template('DF.html', df=df_brasil)

@app.route('/MS')
def MS():

    return render_template('MS.html', df=df_brasil)

@app.route('/GO')
def GO():

    return render_template('GO.html', df=df_brasil)

@app.route('/MG')
def MG():

    return render_template('MG.html', df=df_brasil)

@app.route('/ES')
def ES():

    return render_template('ES.html', df=df_brasil)

@app.route('/RJ')
def RJ():

    return render_template('RJ.html', df=df_brasil)

@app.route('/SP')
def SP():

    return render_template('SP.html', df=df_brasil)

@app.route('/PR')
def PR():

    return render_template('PR.html', df=df_brasil)

@app.route('/SC')
def SC():

    return render_template('SC.html', df=df_brasil)

@app.route('/RS')
def RS():

    return render_template('RS.html', df=df_brasil)
