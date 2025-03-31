# Importamos las bibliotecas necesarias
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr, PackageNotInstalledError

'''
Implementación del Suavizamiento Exponencial Simple

La fórmula utilizada es:
F_{t+1} = α × D_t + (1 - α) × F_t

Donde:
F_{t+1} = Pronóstico exponencialmente suavizado para el siguiente período
F_t     = Pronóstico exponencialmente suavizado para el período actual
D_t     = Demanda real (valor observado) en el período actual
α       = Constante de suavización (entre 0 y 1)
'''

# Activar la conversión automática entre pandas y R
pandas2ri.activate()

# Mostrar la fórmula al inicio del programa
print("\n" + "=" * 80)
print("SUAVIZAMIENTO EXPONENCIAL SIMPLE")
print("=" * 80)
print("\nFórmula:")
print("F_{t+1} = α × D_t + (1 - α) × F_t")
print("\nDonde:")
print("F_{t+1} = Pronóstico exponencialmente suavizado para el siguiente período")
print("F_t     = Pronóstico exponencialmente suavizado para el período actual")
print("D_t     = Demanda real (valor observado) en el período actual")
print("α       = Constante de suavización (entre 0 y 1)")
print("=" * 80 + "\n")

# Verificar que los paquetes de R necesarios estén instalados
try:
    ggplot2 = importr('ggplot2')
    reshape2 = importr('reshape2')
    print("Los paquetes ggplot2 y reshape2 están correctamente instalados.")
except PackageNotInstalledError as e:
    print(f"ERROR: {e}")
    print("\nPara usar este script, necesitas instalar los siguientes paquetes en R:")
    print("  - ggplot2")
    print("  - reshape2")
    print("\nPuedes instalarlos ejecutando los siguientes comandos en R :")
    print('  install.packages("ggplot2")')
    print('  install.packages("reshape2")')
    sys.exit(1)

# Definir los datos en Python
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
alpha = 0.5

print(f"Usando constante de suavizamiento α = {alpha}")
print("\nDatos originales:")
for key, values in data_dict.items():
    print(f"{key}: {values}")

# Convertir diccionario Python a lista de R
r_data_dict = robjects.ListVector({k: robjects.FloatVector(v) for k, v in data_dict.items()})

# Definir el código R para el suavizamiento exponencial
r_code = """
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
      Calculo = character(0)
    )

    # Para el primer periodo, no hay cálculo previo
    calculo_indicador <- rbind(calculo_indicador, data.frame(
      Periodo = 1,
      D_t = observed_values[1],
      F_t = NA,
      F_t_plus_1 = observed_values[1],
      Calculo = "F_1 = D_1 (valor inicial)"
    ))

    # Aplicamos la fórmula del suavizamiento exponencial
    for (t in 2:length(observed_values)) {
      current_forecast <- smoothed_values[t - 1]  # Pronóstico actual (F_t)
      D_t <- observed_values[t]                   # Valor observado actual (D_t)
      next_forecast <- alpha * D_t + (1 - alpha) * current_forecast  # Pronóstico para el siguiente período (F_{t+1})
      smoothed_values <- c(smoothed_values, next_forecast)

      # Almacenar los cálculos detallados
      formula_calculo <- sprintf(
        "F_%d = %.1f × %.1f + (1 - %.1f) × %.1f = %.1f × %.1f + %.1f × %.1f = %.3f + %.3f = %.3f",
        t, alpha, D_t, alpha, current_forecast, 
        alpha, D_t, (1-alpha), current_forecast,
        alpha * D_t, (1-alpha) * current_forecast, next_forecast
      )

      calculo_indicador <- rbind(calculo_indicador, data.frame(
        Periodo = t,
        D_t = D_t,
        F_t = current_forecast,
        F_t_plus_1 = next_forecast,
        Calculo = formula_calculo
      ))
    }

    # Pronosticar para el año 2020
    current_forecast <- smoothed_values[length(smoothed_values)]  # último pronóstico
    D_t <- observed_values[length(observed_values)]  # último valor observado
    next_forecast_2020 <- alpha * D_t + (1 - alpha) * current_forecast
    smoothed_values <- c(smoothed_values, next_forecast_2020)

    # Almacenar los cálculos detallados para el pronóstico 2020
    t <- length(observed_values) + 1
    formula_calculo <- sprintf(
      "F_%d = %.1f × %.1f + (1 - %.1f) × %.1f = %.1f × %.1f + %.1f × %.1f = %.3f + %.3f = %.3f (Pronóstico 2020)",
      t, alpha, D_t, alpha, current_forecast, 
      alpha, D_t, (1-alpha), current_forecast,
      alpha * D_t, (1-alpha) * current_forecast, next_forecast_2020
    )

    calculo_indicador <- rbind(calculo_indicador, data.frame(
      Periodo = t,
      D_t = NA,  # No hay valor observado para 2020
      F_t = current_forecast,
      F_t_plus_1 = next_forecast_2020,
      Calculo = formula_calculo
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
  smoothed_values_melted <- reshape2::melt(smoothed_values_df, id.vars = "Year", variable.name = "Indicator", value.name = "Value")

  # Devolver los resultados como una lista
  return(list(
    smoothed_values_dict = smoothed_values_dict,
    smoothed_values_df = smoothed_values_df,
    smoothed_values_melted = smoothed_values_melted,
    calculo_detallado = calculo_detallado
  ))
}

# Función para crear el gráfico
crear_grafico <- function(smoothed_values_melted) {
  ggplot2::ggplot(data = smoothed_values_melted, ggplot2::aes(x = Year, y = Value, color = Indicator, group = Indicator)) +
    ggplot2::geom_line(linetype = "dashed") +
    ggplot2::geom_point() +
    ggplot2::labs(title = "Pronósticos de Indicadores para el año 2020 utilizando Suavizamiento Exponencial",
         x = "Año", y = "Proporción (%)") +
    ggplot2::theme_minimal() +
    ggplot2::theme(legend.title = ggplot2::element_blank())
}
"""

# Ejecutar el código R
robjects.r(r_code)

# Obtener la función de suavizamiento exponencial de R
r_suavizamiento_exponencial = robjects.r['suavizamiento_exponencial']

# Ejecutar la función con nuestros datos
results = r_suavizamiento_exponencial(r_data_dict, alpha)

# Depuración de la estructura de resultados
print("\nEstructura de resultados:")
print(f"Tipo de results: {type(results)}")
print(f"Número de elementos: {len(results)}")
for i, item in enumerate(results):
    print(f"Elemento {i}: {type(item)}")
    if hasattr(item, 'names'):
        print(f"  Nombres: {item.names}")

# Convertir los resultados de R a Python
smoothed_values_dict = {k: list(v) for k, v in zip(results[0].names, list(results[0]))}
smoothed_values_df = pandas2ri.rpy2py(results[1])
smoothed_values_melted = pandas2ri.rpy2py(results[2])

# Obtener los cálculos detallados con manejo de excepciones mejorado
try:
    calculos_detallados = {}
    for i, k in enumerate(results[3].names):
        try:
            # Acceder a cada elemento individualmente
            calculos_detallados[k] = pandas2ri.rpy2py(results[3][i])
        except Exception as e:
            print(f"Error al convertir cálculos para el indicador {k}: {e}")
except Exception as e:
    print(f"Error general en conversión de cálculos: {e}")
    calculos_detallados = {}

# Crear el gráfico en R y guardarlo con manejo de errores mejorado
try:
    # Importar grDevices de manera más segura
    from rpy2.robjects.packages import importr

    grdevices = importr('grDevices')

    # Especificar una ruta absoluta
    output_file = os.path.join(os.getcwd(), "suavizamiento_exponencial.png")
    print(f"Guardando gráfico en: {output_file}")

    # Abrir el dispositivo gráfico con manejo de errores
    grdevices.png(file=output_file, width=900, height=600)
    r_crear_grafico = robjects.r['crear_grafico']
    r_crear_grafico(results[2])
    grdevices.dev_off()
    print(f"El gráfico se ha guardado en: {output_file}")
except Exception as e:
    print(f"Error al guardar el gráfico de R: {str(e)}")

# Mostrar los resultados en Python
print("\nPronóstico para 2020:")
predictions_2020 = smoothed_values_df[smoothed_values_df['Year'] == 2020].drop(columns=['Year'])
predictions_2020_df = predictions_2020.T.reset_index()
predictions_2020_df.columns = ['Indicador', 'Pronóstico 2020']
print(predictions_2020_df)

# Mostrar cálculos detallados para un indicador de ejemplo (el primero)
if calculos_detallados:
    print("\nCálculos detallados para el indicador:", list(calculos_detallados.keys())[0])
    ejemplo_calculo = calculos_detallados[list(calculos_detallados.keys())[0]]
    for _, row in ejemplo_calculo.iterrows():
        if not pd.isna(row['D_t']):
            print(f"\nPeríodo {int(row['Periodo'])} (Año {2014 + int(row['Periodo'])}):")
            print(f"  Valor real (D_t): {row['D_t']}")
            if not pd.isna(row['F_t']):
                print(f"  Pronóstico actual (F_t): {row['F_t']}")
            print(f"  Pronóstico siguiente (F_{{t+1}}): {row['F_t_plus_1']}")
            print(f"  Cálculo: {row['Calculo']}")
        else:
            print(f"\nPronóstico para 2020:")
            print(f"  Pronóstico actual (F_t): {row['F_t']}")
            print(f"  Pronóstico 2020 (F_{{t+1}}): {row['F_t_plus_1']}")
            print(f"  Cálculo: {row['Calculo']}")

    print("\nPara ver los cálculos detallados de otros indicadores, modifique el código.")

# También podemos mostrar el gráfico usando matplotlib para verlo directamente en Python
# Convertimos el DataFrame largo a un formato adecuado para matplotlib
try:
    plt.figure(figsize=(12, 8))

    # Para cada indicador, graficamos los valores
    for indicator in data_dict.keys():
        indicator_data = smoothed_values_df[['Year', indicator]]
        plt.plot(indicator_data['Year'], indicator_data[indicator], '--o', label=indicator)

    plt.title("Pronósticos de Indicadores para el año 2020 utilizando Suavizamiento Exponencial")
    plt.xlabel("Año")
    plt.ylabel("Proporción (%)")
    plt.legend(title="")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()
except Exception as e:
    print(f"Error al mostrar el gráfico con matplotlib: {str(e)}")