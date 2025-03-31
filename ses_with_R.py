"""
Implementación del Suavizamiento Exponencial Simple

La fórmula utilizada es:
F_{t+1} = α * D_t + (1 - α) * F_t

Donde:
F_{t+1} = Pronóstico exponencialmente suavizado para el siguiente período
F_t     = Pronóstico exponencialmente suavizado para el período actual
D_t     = Demanda real (valor observado) en el período actual
α       = Constante de suavización (entre 0 y 1)
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr, PackageNotInstalledError


# Función para mostrar la introducción del programa
def mostrar_introduccion():
    """Muestra la información inicial sobre el suavizamiento exponencial"""
    print("\n" + "=" * 80)
    print("SUAVIZAMIENTO EXPONENCIAL SIMPLE")
    print("=" * 80)
    print("\nFórmula:")
    print("F_{t+1} = α * D_t + (1 - α) * F_t")
    print("\nDonde:")
    print("F_{t+1} = Pronóstico exponencialmente suavizado para el siguiente período")
    print("F_t     = Pronóstico exponencialmente suavizado para el período actual")
    print("D_t     = Demanda real (valor observado) en el período actual")
    print("α       = Constante de suavización (entre 0 y 1)")
    print("=" * 80 + "\n")


# Función para verificar paquetes de R
def verificar_paquetes_r():
    """Verifica que los paquetes necesarios de R estén instalados"""
    try:
        importr('ggplot2')
        importr('reshape2')
        print("Los paquetes ggplot2 y reshape2 están correctamente instalados.")
        return True
    except PackageNotInstalledError as e:
        print(f"ERROR: {e}")
        print("\nPara usar este script, necesitas instalar los siguientes paquetes en R:")
        print("  - ggplot2")
        print("  - reshape2")
        print("\nPuedes instalarlos ejecutando los siguientes comandos en R :")
        print('  install.packages("ggplot2")')
        print('  install.packages("reshape2")')
        return False


# Función para definir el código R de suavizamiento exponencial
def definir_codigo_r():
    """Define el código R para el suavizamiento exponencial"""
    return """
    suavizamiento_exponencial <- function(data_dict, alpha = 0.5) {
      # Inicializar el diccionario que almacenará los valores suavizados
      smoothed_values_dict <- list()
      # Diccionario para almacenar los cálculos detallados
      calculo_detallado <- list()

      # Recorrer cada indicador en el diccionario original
      for (indicator_name in names(data_dict)) {
        observed_values <- data_dict[[indicator_name]]

        # Inicializar el primer pronóstico como el primer valor observado
        smoothed_values <- c(observed_values[1])  # El primer pronóstico es igual al primer valor observado

        # Almacenar cálculos detallados para este indicador
        calculo_indicador <- data.frame(
          Periodo = numeric(0),
          D_t = numeric(0),
          F_t = numeric(0),
          F_t_plus_1 = numeric(0),
          stringsAsFactors = FALSE  # Importante para manejar strings correctamente
        )

        # Para el primer periodo, no hay cálculo previo
        calculo_indicador <- rbind(calculo_indicador, data.frame(
          Periodo = 1,
          D_t = observed_values[1],
          F_t = NA,
          F_t_plus_1 = observed_values[1],
          stringsAsFactors = FALSE
        ))

        # Aplicamos la fórmula del suavizamiento exponencial
        for (t in 2:length(observed_values)) {
          current_forecast <- smoothed_values[t - 1]  # Pronóstico actual (F_t)
          D_t <- observed_values[t]                   # Valor observado actual (D_t)
          next_forecast <- alpha * D_t + (1 - alpha) * current_forecast  # Pronóstico para el siguiente período (F_{t+1})
          smoothed_values <- c(smoothed_values, next_forecast)

          calculo_indicador <- rbind(calculo_indicador, data.frame(
            Periodo = t,
            D_t = D_t,
            F_t = current_forecast,
            F_t_plus_1 = next_forecast,
            stringsAsFactors = FALSE
          ))
        }

        # Pronosticar para el año 2020
        current_forecast <- smoothed_values[length(smoothed_values)]  # último pronóstico
        D_t <- observed_values[length(observed_values)]  # último valor observado
        next_forecast_2020 <- alpha * D_t + (1 - alpha) * current_forecast
        smoothed_values <- c(smoothed_values, next_forecast_2020)

        calculo_indicador <- rbind(calculo_indicador, data.frame(
          Periodo = length(observed_values) + 1,
          D_t = NA,  # No hay valor observado para 2020
          F_t = current_forecast,
          F_t_plus_1 = next_forecast_2020,
          stringsAsFactors = FALSE
        ))

        # Guardar la lista de pronósticos en el diccionario de pronósticos
        smoothed_values_dict[[indicator_name]] <- smoothed_values
        calculo_detallado[[indicator_name]] <- calculo_indicador
      }

      # Convertir los pronósticos en un DataFrame y agregar los años como una columna
      smoothed_values_df <- as.data.frame(smoothed_values_dict)
      # Usar seq para manejar dinámicamente el número de años basado en los datos
      smoothed_values_df$Year <- seq(2015, 2015 + length(smoothed_values_dict[[1]]) - 1)

      # Para facilitar el gráfico con ggplot2, necesitamos un DataFrame en formato largo
      smoothed_values_melted <- reshape2::melt(
        smoothed_values_df, 
        id.vars = "Year", 
        variable.name = "Indicator", 
        value.name = "Value"
      )

      # Devolver los resultados como una lista
      return(list(
        smoothed_values_dict = smoothed_values_dict,
        smoothed_values_df = smoothed_values_df,
        smoothed_values_melted = smoothed_values_melted,
        calculo_detallado = calculo_detallado
      ))
    }
    """


# Función para ejecutar análisis de suavizamiento exponencial
def ejecutar_suavizamiento_exponencial(data_dict, alpha=0.5):
    """
    Ejecuta el análisis de suavizamiento exponencial usando R a través de rpy2

    Args:
        data_dict: Diccionario con los datos de los indicadores
        alpha: Constante de suavizamiento (entre 0 y 1)

    Returns:
        Tupla con (smoothed_values_dict, smoothed_values_df, calculos_detallados)
    """
    # Activar la conversión automática entre pandas y R
    pandas2ri.activate()

    # Convertir diccionario Python a lista de R
    r_data_dict = robjects.ListVector({k: robjects.FloatVector(v) for k, v in data_dict.items()})

    # Definir y ejecutar el código R
    r_code = definir_codigo_r()
    robjects.r(r_code)

    # Obtener la función de suavizamiento exponencial de R
    r_suavizamiento_exponencial = robjects.r['suavizamiento_exponencial']

    # Ejecutar la función con nuestros datos
    results = r_suavizamiento_exponencial(r_data_dict, alpha)

    # Convertir los resultados de R a Python
    smoothed_values_dict = {k: list(v) for k, v in zip(results[0].names, list(results[0]))}
    smoothed_values_df = pandas2ri.rpy2py(results[1])

    # Convertir los cálculos detallados
    calculos_detallados = {}
    try:
        calculo_detallado_r = results[3]
        for i, k in enumerate(calculo_detallado_r.names):
            calculos_detallados[k] = pandas2ri.rpy2py(calculo_detallado_r[i])
    except Exception as e:
        print(f"Error al convertir cálculos detallados: {e}")

    return smoothed_values_dict, smoothed_values_df, calculos_detallados


# Función para visualizar los resultados
def visualizar_resultados(data_dict, smoothed_values_df):
    """
    Crea y guarda un gráfico de los resultados usando matplotlib

    Args:
        data_dict: Diccionario con los datos originales
        smoothed_values_df: DataFrame con los valores suavizados
    """
    try:
        plt.figure(figsize=(12, 8))

        # Para cada indicador, graficamos los valores
        for indicator in data_dict.keys():
            indicator_data = smoothed_values_df[['Year', indicator]]
            plt.plot(indicator_data['Year'], indicator_data[indicator], '--o', label=indicator)

        plt.title("Pronósticos de Indicadores para el año 2020 utilizando Suavizamiento Exponencial")
        plt.xlabel("Año")
        plt.ylabel("Proporción (%)")
        plt.legend(title="", loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Guardar el gráfico
        output_file = os.path.join(os.getcwd(), "suavizamiento_exponencial.png")
        plt.savefig(output_file)
        print(f"Gráfico guardado en: {output_file}")

        # Mostrar el gráfico
        plt.show()
    except Exception as e:
        print(f"Error al crear el gráfico: {e}")


# Función para mostrar predicciones para 2020
def mostrar_predicciones_2020(smoothed_values_df):
    """
    Muestra la predicción para el año 2020 de cada indicador

    Args:
        smoothed_values_df: DataFrame con los valores suavizados
    """
    try:
        predictions_2020 = smoothed_values_df[smoothed_values_df['Year'] == 2020].drop(columns=['Year'])
        predictions_2020_df = predictions_2020.T.reset_index()
        predictions_2020_df.columns = ['Indicador', 'Pronóstico 2020']
        predictions_2020_df['Pronóstico 2020'] = predictions_2020_df['Pronóstico 2020'].round(2)

        print("\nPronóstico para 2020:")
        print(predictions_2020_df)
    except Exception as e:
        print(f"Error al mostrar pronósticos para 2020: {e}")


# Función para mostrar cálculos detallados
def mostrar_calculos_detallados(calculos_detallados):
    """
    Muestra los cálculos detallados para un indicador de ejemplo

    Args:
        calculos_detallados: Diccionario con los cálculos detallados por indicador
    """
    if not calculos_detallados:
        print("No hay cálculos detallados disponibles.")
        return

    try:
        # Tomar el primer indicador como ejemplo
        indicator_key = list(calculos_detallados.keys())[0]
        print(f"\nCálculos detallados para el indicador: {indicator_key}")

        ejemplo_calculo = calculos_detallados[indicator_key]
        for _, row in ejemplo_calculo.iterrows():
            periodo = int(row['Periodo'])
            if not pd.isna(row['D_t']):
                print(f"\nPeríodo {periodo} (Año {2014 + periodo}):")
                print(f"  Valor real (D_t): {row['D_t']}")
                if not pd.isna(row['F_t']):
                    print(f"  Pronóstico actual (F_t): {row['F_t']}")
                print(f"  Pronóstico siguiente (F_t+1): {row['F_t_plus_1']}")
                print(
                    f"  Cálculo: F_{periodo + 1} = {row['F_t_plus_1']} = {alpha} * {row['D_t']} + (1 - {alpha}) * {row['F_t']}")
            else:
                print(f"\nPronóstico para 2020:")
                print(f"  Pronóstico actual (F_t): {row['F_t']}")
                print(f"  Pronóstico 2020 (F_t+1): {row['F_t_plus_1']}")

        print("\nPara ver los cálculos detallados de otros indicadores, modifique el código.")
    except Exception as e:
        print(f"Error al mostrar cálculos detallados: {e}")


# Función principal
def main():
    # Mostrar información inicial
    mostrar_introduccion()

    # Verificar paquetes de R
    if not verificar_paquetes_r():
        sys.exit(1)

    # Definir los datos
    data_dict = {
        'HC': [44.9, 45.6, 45.4, 44.9, 44.3],
        'HI': [39.2, 47, 50.9, 52.9, 56.4],
        'HT': [93.5, 93.1, 93.2, 92.9, 92.5],
        'HTP': [43.7, 52.1, 49.5, 47.3, 45.9],
        'U6E': [51.3, 47, 45.3, 45, 43],
        'UI6E': [57.4, 59.5, 63.9, 65.8, 70.1],
        'UCHE': [51.3, 52.2, 46.8, 46.7, 44.6],
        'UITI': [12.8, 14.7, 20.4, 23.7, 27.2],
        'UIFH': [29.1, 20.5, 16.7, 13.4, 10.7],
        'UTC6E': [71.6, 73.6, 72.2, 73.5, 75.1]
    }

    # Definir alpha (constante de suavizamiento)
    global alpha
    alpha = 0.5

    print(f"Usando constante de suavizamiento α = {alpha}")
    print("\nDatos originales:")
    for key, values in data_dict.items():
        print(f"{key}: {values}")

    # Ejecutar el suavizamiento exponencial
    smoothed_values_dict, smoothed_values_df, calculos_detallados = ejecutar_suavizamiento_exponencial(
        data_dict, alpha
    )

    # Mostrar predicciones para 2020
    mostrar_predicciones_2020(smoothed_values_df)

    # Mostrar cálculos detallados para un indicador de ejemplo
    mostrar_calculos_detallados(calculos_detallados)

    # Visualizar los resultados
    visualizar_resultados(data_dict, smoothed_values_df)


# Ejecutar el programa si se llama directamente
if __name__ == "__main__":
    main()