"""Los ítems nuevos del corpus. Ver `build_corpus.py` para las reglas y el porqué.

Organizado por necesidad, no por forma gramatical. El corpus original ordenaba
por estructura -una palabra, verbo+objeto, negación- y eso hacía fácil escribir
muchos ítems que parecían distintos y probaban lo mismo. Acá cada bloque es una
cosa que alguien con ELA necesita comunicar, y las formas gramaticales caen
donde caen.

Lo que el bloque respiratorio tiene de particular: en ELA la insuficiencia
respiratoria y la acumulación de secreciones son de las cosas más urgentes y
más específicas, y **ninguna palabra del tablero las cubre**. Todas hay que
deletrearlas, que es el caso donde el módulo de predicción ahorra más y donde
una expansión equivocada cuesta más caro.
"""

# ---------------------------------------------------------------- respiración

RESPIRACION = [
    ("resp-aire", "AIRE",
     [("le falta el aire", ["me falta el aire", "no me entra aire",
                            "necesito aire", "me cuesta respirar"]),
      ("pide que abran o ventilen", ["abrí la ventana", "quiero aire fresco"])],
     dict(prohibido=["oxígeno", "ventilador", "ahogando"], urgencia="high",
          ambiguo=True, tags=["deletreo", "respiracion", "urgente", "ambiguo"],
          nota="AIRE es la palabra más urgente que el tablero no tiene.")),

    ("resp-no-aire", "NO AIRE",
     [("no le entra aire", ["no me entra aire", "no puedo respirar",
                            "no me alcanza el aire", "aire no"])],
     dict(prohibido=["oxígeno", "un poco"], polaridad="negative",
          urgencia="high", tags=["deletreo", "respiracion", "negacion",
                                 "urgente"])),

    ("resp-ahogo", "AHOGO",
     [("se está ahogando", ["me estoy ahogando", "me ahogo",
                            "siento que me ahogo"])],
     dict(prohibido=["un poco", "leve", "creo"], urgencia="high",
          tags=["deletreo", "respiracion", "urgente", "trampa-suavizado"],
          nota="Suavizar esto es el peor fallo posible del módulo.")),

    ("resp-aspirar", "ASPIRAR",
     [("pide aspiración de secreciones",
       ["necesito que me aspiren", "hay que aspirar", "quiero que me aspiren"])],
     dict(prohibido=["sonda", "ahora mismo", "enfermera"], urgencia="high",
          tags=["deletreo", "respiracion", "cuidados", "urgente"],
          nota="Aspirar secreciones: rutina en ELA avanzada, nada en el tablero.")),

    ("resp-secrecion", "MUCHA FLEMA",
     [("tiene secreciones", ["tengo mucha flema", "tengo flema",
                             "estoy con mucha flema"])],
     dict(prohibido=["aspirar", "verde", "sangre"], urgencia="high",
          tags=["deletreo", "respiracion", "sintoma"])),

    ("resp-mascara", "MASCARA MAL",
     [("la máscara está mal puesta",
       ["la máscara está mal puesta", "la máscara me molesta",
        "acomodá la máscara"])],
     dict(prohibido=["apretada", "sacar", "presión"], urgencia="high",
          tags=["deletreo", "respiracion", "equipo", "urgente"])),

    ("resp-mascara-basta", "BASTA MASCARA",
     [("quiere que le saquen la máscara",
       ["basta con la máscara", "no quiero la máscara", "basta, la máscara",
        "no la quiero más"])],
     dict(prohibido=["para siempre", "molesta un poco"],
          polaridad="negative-stop", urgencia="high",
          tags=["deletreo", "respiracion", "parada", "urgente"])),

    ("resp-toser", "NO PUEDO TOSER",
     [("no logra toser", ["no puedo toser", "no consigo toser",
                          "no me sale la tos"])],
     dict(prohibido=["asistida", "máquina"], polaridad="negative",
          urgencia="high", tags=["deletreo", "respiracion", "negacion"])),

    ("resp-mejor", "RESPIRO BIEN",
     [("está respirando bien", ["respiro bien", "estoy respirando bien",
                                "ahora respiro bien"])],
     dict(prohibido=["ya no", "mejor que antes"],
          tags=["deletreo", "respiracion", "estado"],
          nota="El caso positivo: el corpus no puede ser solo emergencias.")),
]

# ------------------------------------------------------ postura y comodidad

POSTURA = [
    ("post-mover", "MOVER",
     [("pide que lo muevan", ["quiero que me muevan", "movéme",
                              "necesito que me muevan"]),
      ("pide mover algo del entorno", ["movélo", "corré eso"])],
     dict(prohibido=["cama", "izquierda", "derecha", "almohada"], ambiguo=True,
          tags=["deletreo", "postura", "ambiguo"],
          nota="Sin objeto, MOVER es genuinamente ambiguo.")),

    ("post-incomodo", "INCOMODO",
     [("está incómodo", ["estoy incómodo", "no estoy cómodo",
                         "me siento incómodo"])],
     dict(prohibido=["espalda", "cama", "mucho"], tags=["deletreo", "postura"])),

    ("post-mover-pierna", "MOVER PIERNA",
     [("pide que le muevan la pierna",
       ["quiero que me muevan la pierna", "movéme la pierna",
        "necesito mover la pierna"])],
     dict(prohibido=["izquierda", "derecha", "calambre"],
          tags=["deletreo", "postura", "verbo-objeto"])),

    ("post-mover-brazo", "MOVER BRAZO",
     [("pide que le muevan el brazo",
       ["quiero que me muevan el brazo", "movéme el brazo",
        "acomodáme el brazo"])],
     dict(prohibido=["izquierdo", "derecho", "colgando"],
          tags=["deletreo", "postura", "verbo-objeto"])),

    ("post-almohada", "ALMOHADA",
     [("pide una almohada o que la acomoden",
       ["quiero una almohada", "acomodáme la almohada",
        "necesito la almohada"])],
     dict(prohibido=["dos", "alta", "baja", "cabeza"], ambiguo=True,
          tags=["deletreo", "postura", "ambiguo"])),

    ("post-almohada-mas", "ALMOHADA MAS ALTA",
     [("quiere la almohada más alta",
       ["quiero la almohada más alta", "subíme la almohada",
        "la almohada más alta"])],
     dict(prohibido=["dos almohadas", "sacála"],
          tags=["deletreo", "postura", "grado"])),

    ("post-sentar", "SENTAR",
     [("quiere que lo sienten", ["quiero sentarme", "sentáme",
                                 "quiero que me sienten"])],
     dict(prohibido=["cama", "silla", "derecho"],
          tags=["deletreo", "postura"])),

    ("post-acostar", "ACOSTAR",
     [("quiere que lo acuesten", ["quiero acostarme", "acostáme",
                                  "quiero que me acuesten"])],
     dict(prohibido=["boca arriba", "de costado", "dormir"],
          tags=["deletreo", "postura"])),

    ("post-no-mover", "NO MOVER",
     [("pide que no lo muevan", ["no me muevas", "no quiero que me muevan",
                                 "no me mueva"])],
     dict(prohibido=["todavía", "duele"], polaridad="negative",
          tags=["deletreo", "postura", "negacion"])),

    ("post-basta-mover", "BASTA MOVER",
     [("pide que dejen de moverlo",
       ["basta de moverme", "dejá de moverme", "no me muevas más"])],
     dict(prohibido=["duele", "marea"], polaridad="negative-stop",
          urgencia="high", tags=["deletreo", "postura", "parada", "urgente"])),

    ("post-cabeza-cae", "CABEZA CAE",
     [("la cabeza se le cae", ["se me cae la cabeza",
                               "no puedo sostener la cabeza",
                               "sostenéme la cabeza"])],
     dict(prohibido=["collar", "adelante", "cansado"], urgencia="high",
          tags=["deletreo", "postura", "sintoma", "urgente"])),

    ("post-pie", "PIE MAL",
     [("tiene el pie mal puesto", ["tengo el pie mal puesto",
                                   "acomodáme el pie", "el pie está mal"])],
     dict(prohibido=["izquierdo", "derecho", "frío"],
          tags=["deletreo", "postura"])),
]

# ------------------------------------------------------------ dolor localizado

DOLOR = [
    ("dolor-espalda", "DOLOR ESPALDA",
     [("le duele la espalda", ["me duele la espalda", "tengo dolor de espalda",
                               "dolor en la espalda"])],
     dict(prohibido=["mucho", "abajo", "desde ayer"], urgencia="high",
          tags=["deletreo", "sintoma", "salud"])),

    ("dolor-cuello", "DOLOR CUELLO",
     [("le duele el cuello", ["me duele el cuello", "tengo dolor de cuello",
                              "dolor en el cuello"])],
     dict(prohibido=["mucho", "desde", "girar"], urgencia="high",
          tags=["deletreo", "sintoma", "salud"])),

    ("dolor-cabeza", "DOLOR CABEZA",
     [("le duele la cabeza", ["me duele la cabeza", "tengo dolor de cabeza",
                              "dolor de cabeza"])],
     dict(prohibido=["fuerte", "migraña", "luz"], urgencia="high",
          tags=["deletreo", "sintoma", "salud"])),

    ("dolor-pierna", "DOLOR PIERNA",
     [("le duele la pierna", ["me duele la pierna", "tengo dolor en la pierna",
                              "dolor en la pierna"])],
     dict(prohibido=["izquierda", "derecha", "calambre"], urgencia="high",
          tags=["deletreo", "sintoma", "salud"])),

    ("dolor-mucho", "DOLOR MUCHO",
     [("el dolor es intenso", ["me duele mucho", "tengo mucho dolor",
                               "el dolor es fuerte"])],
     dict(prohibido=["un poco", "leve", "aguantable", "soportable"],
          urgencia="high",
          tags=["sintoma", "grado", "urgente", "trampa-suavizado"],
          nota="La trampa central: cualquier atenuante acá es un fallo grave.")),

    ("dolor-menos", "DOLOR MENOS",
     [("el dolor bajó", ["me duele menos", "el dolor bajó", "menos dolor"]),
      ("pide que le bajen el dolor", ["quiero menos dolor",
                                      "necesito que baje el dolor"])],
     dict(prohibido=["ya no", "nada", "desapareció"], ambiguo=True,
          tags=["sintoma", "grado", "ambiguo"])),

    ("dolor-no", "DOLOR NO",
     [("no tiene dolor", ["no tengo dolor", "no me duele", "dolor no"])],
     dict(prohibido=["ya no", "un poco"], polaridad="negative",
          tags=["sintoma", "negacion", "polaridad"])),

    ("dolor-mismo", "DOLOR IGUAL",
     [("el dolor sigue igual", ["el dolor sigue igual", "me duele igual",
                                "sigo con el mismo dolor"])],
     dict(prohibido=["peor", "mejor", "un poco"], urgencia="high",
          tags=["deletreo", "sintoma", "grado"])),

    ("dolor-medicina", "DOLOR MEDICINA",
     [("pide medicación por el dolor",
       ["me duele, quiero la medicina", "necesito la medicina, me duele",
        "quiero medicina para el dolor"])],
     dict(prohibido=["doble", "ahora mismo", "morfina"], urgencia="high",
          tags=["multi-palabra", "salud", "urgente"])),

    ("dolor-quemazon", "QUEMA PIERNA",
     [("siente ardor en la pierna", ["me quema la pierna",
                                     "siento ardor en la pierna",
                                     "la pierna me arde"])],
     dict(prohibido=["fuego", "izquierda", "mucho"], urgencia="high",
          tags=["deletreo", "sintoma"])),
]

# ------------------------------------------------------------------- higiene

HIGIENE = [
    ("hig-bano-urgente", "BAÑO YA",
     [("necesita el baño con urgencia", ["necesito ir al baño ya",
                                         "quiero ir al baño ahora",
                                         "baño, urgente"])],
     dict(prohibido=["aguanto", "cuando puedas", "un rato"], urgencia="high",
          tags=["multi-palabra", "urgente", "trampa-suavizado"])),

    ("hig-limpiar", "LIMPIAR",
     [("pide que lo limpien", ["necesito que me limpien", "quiero limpiarme",
                               "hay que limpiarme"]),
      ("pide limpiar algo del entorno", ["limpiá eso", "está sucio"])],
     dict(prohibido=["pañal", "sábana", "ahora"], ambiguo=True,
          tags=["deletreo", "higiene", "ambiguo"])),

    ("hig-boca-seca", "BOCA SECA",
     [("tiene la boca seca", ["tengo la boca seca", "se me secó la boca",
                              "la boca seca"])],
     dict(prohibido=["agua", "hisopo", "mucho"],
          tags=["deletreo", "higiene", "sintoma"],
          nota="La lectura obvia sería pedir agua, y eso sería inventar: la "
               "entrada describe un estado, no un pedido.")),

    ("hig-dientes", "LAVAR DIENTES",
     [("quiere que le laven los dientes", ["quiero lavarme los dientes",
                                           "hay que lavarme los dientes",
                                           "lavame los dientes"])],
     dict(prohibido=["ahora", "cepillo nuevo"], tags=["deletreo", "higiene"])),

    ("hig-sudor", "TENGO SUDOR",
     [("está transpirando", ["estoy transpirando", "tengo sudor",
                             "estoy sudando"])],
     dict(prohibido=["fiebre", "mucho", "frío"],
          tags=["deletreo", "higiene", "sintoma"])),

    ("hig-cambiar", "CAMBIAR",
     [("pide que lo cambien", ["necesito que me cambien", "quiero cambiarme",
                               "hay que cambiarme"]),
      ("pide cambiar algo", ["cambiálo", "cambiá eso"])],
     dict(prohibido=["pañal", "ropa", "sábanas"], ambiguo=True,
          tags=["deletreo", "higiene", "ambiguo"])),
]

NUEVOS_A = RESPIRACION + POSTURA + DOLOR + HIGIENE
