import matplotlib.pyplot as plt
import numpy as np

def set_plot_style():

    plt.rcParams.update({
        'axes.grid': False,  # Sem linhas de grade
        'axes.linewidth': 1.5,  # Largura da linha dos eixos
        'axes.edgecolor': 'black',  # Cor da linha dos eixos
        'axes.facecolor': 'white',  # Sem preenchimento
        'figure.facecolor': 'white',  # Sem preenchimento da figura
        'axes.spines.top': False,  # Sem linha superior
        'axes.spines.right': False,  # Sem linha direita
        'axes.titlesize': 0,  # Sem título do gráfico
        'axes.titlepad': -10,
        'axes.labelsize': 11,  # Tamanho da fonte dos títulos dos eixos
        'font.family': 'Arial',  # Fonte Arial
        'font.size': 11,  # Tamanho da fonte geral (se necessário)
        'axes.labelcolor': 'black',  # Cor da fonte dos títulos dos eixos
        'xtick.major.size': 0,  # Comprimento dos ticks principais do eixo x
        'ytick.major.size': 0,  # Comprimento dos ticks principais do eixo y
        'xtick.minor.size': 0,  # Comprimento dos ticks menores do eixo x (se houver)
        'ytick.minor.size': 0,
        'xtick.major.pad': 10,   # Espaçamento entre o tick e o label do eixo x (em pontos)
        'ytick.major.pad': 10 
    })

    print("Ambiente configurado com sucesso.")