import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
Implementación de suavizamiento exponencial simple

La fórmula implementada es:
S_t = α × Y_t + (1 - α) × S_(t-1)

Donde:
- S_t = Pronóstico exponencialmente suavizado para el período actual
- S_(t-1) = Pronóstico exponencialmente suavizado para el período anterior
- Y_t = Demanda real (valor observado) en el período actual
- α = Constante de suavización (valor entre 0 y 1)
'''

# Estructura de los datos
data_dict = {
    'HC': [44.9, 45.6, 45.4, 44.9, 44.3],
    'HI': [39.2, 47.0, 50.9, 52.9, 56.4],
    'HT': [93.5, 93.1, 93.2, 92.9, 92.5],
    'HTP': [43.7, 52.1, 49.5, 47.3, 45.9],
    'U6E': [51.3, 47.0, 45.3, 45.0, 43.0],
    'UI6E': [57.4, 59.5, 63.9, 65.8, 70.1],
    'UCHE': [51.3, 52.2, 46.8, 46.7, 44.6],
    'UITI': [12.8, 14.7, 20.4, 23.7, 27.2],
    'UIFH': [29.1, 20.5, 16.7, 13.4, 10.7],
    'UTC6E': [71.6, 73.6, 72.2, 73.5, 75.1]
}
years = [2015, 2016, 2017, 2018, 2019]


def aplicar_suavizamiento_exponencial(valores, alpha):
    """
    Aplica suavizamiento exponencial simple a una serie temporal

    Implementa la fórmula: S_t = α × Y_t + (1 - α) × S_(t-1)

    Args:
        valores (list): Lista de valores observados (Y_t)
        alpha (float): Constante de suavizamiento (α) entre 0 y 1

    Returns:
        tuple: (valores_suavizados, pronóstico_siguiente, métricas_error)
    """
    observed = np.array(valores)  # Y_t (valores observados)
    n = len(observed)
    smoothed = np.zeros(n)  # S_t (valores suavizados)

    # Inicializar con el primer valor observado
    # S_0 = Y_0 (el primer pronóstico es igual al primer valor observado)
    smoothed[0] = observed[0]

    # Aplicar la fórmula de suavizamiento exponencial para cada período
    for t in range(1, n):
        # S_t = α × Y_t + (1 - α) × S_(t-1)
        smoothed[t] = alpha * observed[t] + (1 - alpha) * smoothed[t - 1]

    # Pronosticar el siguiente período (t+1) usando la misma fórmula
    # S_(t+1) = α × Y_t + (1 - α) × S_t
    next_forecast = alpha * observed[-1] + (1 - alpha) * smoothed[-1]

    # Calcular errores para evaluar la precisión del modelo
    # El error es la diferencia entre el valor observado y el pronóstico del período anterior
    errors = observed[1:] - smoothed[:-1]  # Error de pronóstico

    # Calcular métricas de error comunes
    mse = np.mean(errors ** 2)  # Error cuadrático medio
    rmse = np.sqrt(mse)  # Raíz del error cuadrático medio
    mae = np.mean(np.abs(errors))  # Error absoluto medio

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae
    }

    return smoothed, next_forecast, metrics


def encontrar_alpha_optimo(valores, alpha_range=np.arange(0.1, 1.0, 0.1)):
    """
    Encuentra el valor de alpha que minimiza el error de pronóstico

    Args:
        valores (list): Lista de valores observados
        alpha_range (array): Rango de valores de alpha a probar

    Returns:
        tuple: (alpha_óptimo, métricas_con_alpha_óptimo)
    """
    min_mse = float('inf')
    best_alpha = None
    best_metrics = None

    # Probar diferentes valores de alpha y seleccionar el que produce el menor error
    for alpha in alpha_range:
        _, _, metrics = aplicar_suavizamiento_exponencial(valores, alpha)
        if metrics['MSE'] < min_mse:
            min_mse = metrics['MSE']
            best_alpha = alpha
            best_metrics = metrics

    return best_alpha, best_metrics


def procesar_indicadores(data_dict, years, optimize=True, fixed_alpha=0.5):
    """
    Procesa todos los indicadores y genera pronósticos

    Args:
        data_dict (dict): Diccionario con los valores observados por indicador
        years (list): Lista de años correspondientes
        optimize (bool): Si es True, encuentra el alpha óptimo para cada indicador
        fixed_alpha (float): Valor de alpha fijo si optimize es False

    Returns:
        tuple: (DataFrame de resultados, DataFrame de alphas y métricas)
    """
    smoothed_values_dict = {}
    optimal_alphas = {}
    error_metrics = {}

    # Procesar cada indicador en el diccionario
    for indicator, values in data_dict.items():
        if optimize:
            # Encontrar el alpha óptimo para este indicador
            best_alpha, best_metrics = encontrar_alpha_optimo(values)
            optimal_alphas[indicator] = best_alpha
            error_metrics[indicator] = best_metrics

            # Aplicar suavizamiento con el alpha óptimo
            smoothed, forecast_next, _ = aplicar_suavizamiento_exponencial(values, best_alpha)
        else:
            # Usar alpha fijo (igual para todos los indicadores)
            smoothed, forecast_next, metrics = aplicar_suavizamiento_exponencial(values, fixed_alpha)
            optimal_alphas[indicator] = fixed_alpha
            error_metrics[indicator] = metrics

        # Guardar los resultados (valores suavizados + pronóstico)
        smoothed_values = list(smoothed) + [forecast_next]
        smoothed_values_dict[indicator] = smoothed_values

    # Crear DataFrame con resultados
    results_df = pd.DataFrame(smoothed_values_dict)
    results_df.index = years + [years[-1] + 1]  # Añadir año de pronóstico
    results_df.index.name = 'Year'

    # Crear tabla de alphas y métricas de error
    alpha_df = pd.DataFrame({
        'Optimal Alpha': optimal_alphas,
        'MSE': {k: v['MSE'] for k, v in error_metrics.items()},
        'RMSE': {k: v['RMSE'] for k, v in error_metrics.items()},
        'MAE': {k: v['MAE'] for k, v in error_metrics.items()}
    })

    return results_df, alpha_df


def visualizar_pronosticos(results_df, years):
    """
    Genera el gráfico de líneas de los pronósticos para todos los indicadores

    Args:
        results_df (DataFrame): DataFrame con los valores suavizados y pronósticos
        years (list): Lista de años utilizados en el análisis

    Returns:
        matplotlib.figure.Figure: Figura generada con el gráfico de pronósticos
    """
    # Configurar estilo visual
    sns.set_style("whitegrid")
    sns.set_palette("tab10")

    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(14, 8))

    # Convertir a formato largo para seaborn
    results_melted = results_df.reset_index().melt(id_vars="Year",
                                                   var_name="Indicator",
                                                   value_name="Value")

    # Gráfico de líneas
    sns.lineplot(data=results_melted, x='Year', y='Value', hue='Indicator',
                 style='Indicator', markers=True, dashes=False, linewidth=2.5, ax=ax)

    # Destacar el pronóstico
    forecast_year = years[-1] + 1
    forecast_data = results_melted[results_melted['Year'] == forecast_year]
    sns.scatterplot(data=forecast_data, x='Year', y='Value', hue='Indicator',
                    style='Indicator', s=100, ax=ax, legend=False, edgecolor='black')

    # Personalizar el gráfico
    ax.set_title(f"Pronósticos de Indicadores para {forecast_year} con Suavizamiento Exponencial",
                 fontsize=16, fontweight='bold')
    ax.set_xlabel("Año", fontsize=12)
    ax.set_ylabel("Valor del Indicador (%)", fontsize=12)
    ax.set_xticks(years + [forecast_year])
    ax.grid(True, linestyle='--', alpha=0.7)

    # Ajustar la leyenda
    plt.legend(title='Indicadores', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    plt.tight_layout()

    return fig


def visualizar_alphas(alpha_df):
    """
    Genera el gráfico de barras de los valores óptimos de alpha por indicador

    Args:
        alpha_df (DataFrame): DataFrame con los alphas óptimos y métricas

    Returns:
        matplotlib.figure.Figure: Figura generada con el gráfico de alphas
    """
    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Gráfico de barras
    sns.barplot(x=alpha_df.index, y='Optimal Alpha', data=alpha_df, ax=ax)

    # Personalizar el gráfico
    ax.set_title("Valores Óptimos de Alpha por Indicador", fontsize=14, fontweight='bold')
    ax.set_xlabel("Indicador", fontsize=12)
    ax.set_ylabel("Alpha Óptimo", fontsize=12)
    plt.xticks(rotation=45)

    # Añadir los valores sobre las barras
    for i, v in enumerate(alpha_df['Optimal Alpha']):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=9)

    plt.tight_layout()

    return fig


def visualizar_resultados(results_df, alpha_df, years, mostrar_pronosticos=True, mostrar_alphas=True):
    """
    Genera los gráficos seleccionados con los resultados del análisis

    Esta función coordina la generación de visualizaciones según los parámetros
    booleanos que indican qué gráficos se deben crear.

    Args:
        results_df (DataFrame): DataFrame con los valores suavizados y pronósticos
        alpha_df (DataFrame): DataFrame con los alphas óptimos y métricas
        years (list): Lista de años utilizados en el análisis
        mostrar_pronosticos (bool): Si es True, genera el gráfico de pronósticos
        mostrar_alphas (bool): Si es True, genera el gráfico de alphas óptimos

    Returns:
        tuple: Contiene las figuras generadas (None para las no generadas)
              (fig_pronosticos, fig_alphas)
    """
    fig_pronosticos = None
    fig_alphas = None

    # Generar gráfico de pronósticos si se solicita
    if mostrar_pronosticos:
        fig_pronosticos = visualizar_pronosticos(results_df, years)
        print("Gráfico de pronósticos generado.")

    # Generar gráfico de alphas óptimos si se solicita
    if mostrar_alphas:
        fig_alphas = visualizar_alphas(alpha_df)
        print("Gráfico de alphas óptimos generado.")

    return fig_pronosticos, fig_alphas

def imprimir_resumen(results_df, alpha_df, years):
    """
    Imprime un resumen de los resultados

    Args:
        results_df (DataFrame): DataFrame con los valores suavizados y pronósticos
        alpha_df (DataFrame): DataFrame con los alphas óptimos y métricas
        years (list): Lista de años utilizados en el análisis
    """
    forecast_year = years[-1] + 1

    print(f"\nProyecciones para {forecast_year}:")
    print(results_df.loc[forecast_year].sort_values(ascending=False))

    print("\nValores de Alpha Óptimos y Métricas de Error:")
    print(alpha_df)


# Ahora modificamos la función de análisis completo para incluir estos parámetros
def realizar_analisis_completo(data_dict, years, optimize=True, fixed_alpha=0.5,
                               mostrar_pronosticos=True, mostrar_alphas=True):
    """
    Realiza el análisis completo de suavizamiento exponencial

    Esta función coordina el flujo completo de trabajo:
    1. Procesa los indicadores para obtener pronósticos
    2. Visualiza los resultados en gráficos (según los parámetros)
    3. Imprime un resumen de los resultados

    Args:
        data_dict (dict): Diccionario con datos por indicador
        years (list): Lista de años correspondientes
        optimize (bool): Si es True, encuentra el alpha óptimo para cada indicador
        fixed_alpha (float): Valor de alpha fijo si optimize es False
        mostrar_pronosticos (bool): Si es True, genera el gráfico de pronósticos
        mostrar_alphas (bool): Si es True, genera el gráfico de alphas óptimos

    Returns:
        tuple: (results_df, alpha_df, figuras) - DataFrames y figuras generadas
    """
    # 1. PROCESAMIENTO DE DATOS
    print("Iniciando procesamiento de datos...")
    results_df, alpha_df = procesar_indicadores(data_dict, years, optimize, fixed_alpha)
    print("Procesamiento completado.")

    # 2. VISUALIZACIÓN DE RESULTADOS (condicional)
    print("Generando visualizaciones solicitadas...")
    figuras = visualizar_resultados(results_df, alpha_df, years,
                                    mostrar_pronosticos, mostrar_alphas)

    # 3. PRESENTACIÓN DE RESULTADOS
    # ---------------------------
    # Esta etapa muestra un resumen de los pronósticos y métricas
    print("Preparando resumen de resultados...")
    imprimir_resumen(results_df, alpha_df, years)
    print("Análisis completo finalizado.")

    return results_df, alpha_df, figuras


# ======= CÓDIGO PRINCIPAL =======
# Ejemplo de uso en el bloque principal
if __name__ == "__main__":
    """
    Punto de entrada principal del programa
    """
    print("=== ANÁLISIS DE SUAVIZAMIENTO EXPONENCIAL ===")
    print("Iniciando análisis...")

    # Definir qué gráficos mostrar
    generar_grafico_pronosticos = True  # Cambiar a False para no generar este gráfico
    generar_grafico_alphas = False  # Cambiar a False para no generar este gráfico

    # Realizar el análisis completo con los parámetros de visualización
    resultados, alphas, (fig_pronosticos, fig_alphas) = realizar_analisis_completo(
        data_dict,
        years,
        optimize=True,
        mostrar_pronosticos=generar_grafico_pronosticos,
        mostrar_alphas=generar_grafico_alphas
    )

    # Mostrar los gráficos generados
    print("\nMostrando gráficos generados...")
    plt.show()  # Esto mostrará solo los gráficos que se generaron

    print("\nPrograma finalizado.")