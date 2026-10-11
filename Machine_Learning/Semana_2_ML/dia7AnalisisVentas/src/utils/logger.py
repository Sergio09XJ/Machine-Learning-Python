import logging

def obtener_logger(nombre_modulo):
    logger = logging.getLogger(nombre_modulo) #Crea una istancia para los logs, esto para cada arc    hivo. 
   
    if not logger.handlers: #Esto evita que se creen varias istancias por modulo/archivo. 
      logger.setLevel(logging.INFO) #Indica el nivel del log DEBUG -> INFO -> WARNING -> ERROR
   
      formato = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s") #Forma    to de salida de log. 
  
      archivo_handler = logging.FileHandler("app.log") #Indicamos donde se guardaran los logs. 
      archivo_handler.setFormatter(formato) #Indicamos el formato. 
  
      logger.addHandler(archivo_handler) #Conecta el logger con el archivo. 
  
    return logger