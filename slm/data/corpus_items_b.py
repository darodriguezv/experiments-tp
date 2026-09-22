"""Corpus, segundo bloque: deglución, temperatura, personas y entorno.

Separado por tema y no por forma, igual que el primero. Ver `corpus_items.py`
para el criterio y `build_corpus.py` para las reglas que todo ítem debe cumplir.
"""

# ------------------------------------------------------- deglución y comida

DEGLUCION = [
    ("deg-no-tragar", "NO PUEDO TRAGAR",
     [("no logra tragar", ["no puedo tragar", "no consigo tragar",
                           "no me pasa"])],
     dict(prohibido=["sonda", "atragantar", "liquido"], polaridad="negative",
          urgencia="high",
          tags=["deletreo", "deglucion", "negacion", "urgente"])),

    ("deg-atragantar", "ME ATRAGANTO",
     [("se está atragantando", ["me atraganté", "me estoy atragantando",
                                "se me fue por otro lado"])],
     dict(prohibido=["casi", "leve"], urgencia="high",
          tags=["deletreo", "deglucion", "urgente", "trampa-suavizado"])),

    ("deg-despacio", "DESPACIO",
     [("pide que vayan más lento", ["más despacio", "despacio, por favor",
                                    "andá más lento"])],
     dict(prohibido=["cuchara", "hablar"], ambiguo=True,
          tags=["deletreo", "ritmo", "ambiguo"],
          nota="Puede referirse a la comida, al habla o a cualquier cosa que "
               "esté pasando. El contexto no está en la entrada, así que "
               "elegir uno sería inventarlo.")),

    ("com-comida-basta", "BASTA COMIDA",
     [("no quiere más comida", ["basta de comida", "no quiero más comida",
                                "ya no quiero comer", "basta, comida"])],
     dict(prohibido=["fria", "no me gusta"], polaridad="negative-stop",
          tags=["multi-palabra", "parada", "polaridad"])),

    ("com-comida-mas", "COMIDA MAS",
     [("quiere más comida", ["quiero más comida", "más comida por favor",
                             "quiero comer más"])],
     dict(prohibido=["postre", "sal", "caliente"],
          tags=["multi-palabra", "grado", "orden-invertido"])),

    ("com-comida-fria", "COMIDA FRIA",
     [("la comida está fría", ["la comida está fría",
                               "esta comida está fría", "comida fría"])],
     dict(prohibido=["calentala", "asquerosa", "tirala"],
          tags=["multi-palabra", "descripcion"],
          nota="Describe un hecho. Convertirlo en una orden -«calentála»- "
               "sería inventar un pedido que la entrada no hace.")),

    ("com-no-hambre", "NO HAMBRE",
     [("no tiene hambre", ["no tengo hambre", "no quiero comer",
                           "hambre no"])],
     dict(prohibido=["despues", "nauseas"], polaridad="negative",
          tags=["multi-palabra", "negacion"])),

    ("com-sonda", "SONDA MAL",
     [("la sonda está mal", ["la sonda está mal", "la sonda me molesta",
                             "algo anda mal con la sonda"])],
     dict(prohibido=["salio", "sangre", "tapada"], urgencia="high",
          tags=["deletreo", "equipo", "urgente"])),
]

# ---------------------------------------------------------------- temperatura

TEMPERATURA = [
    ("temp-frio", "FRIO",
     [("tiene frío", ["tengo frío", "siento frío", "estoy con frío"]),
      ("pide algo para el frío", ["quiero una manta", "tapame"])],
     dict(prohibido=["calefaccion", "temblando"], ambiguo=True,
          tags=["deletreo", "temperatura", "ambiguo"])),

    ("temp-calor", "CALOR",
     [("tiene calor", ["tengo calor", "siento calor", "estoy con calor"]),
      ("pide algo para el calor", ["destapame", "prendé el ventilador"])],
     dict(prohibido=["fiebre", "aire acondicionado"], ambiguo=True,
          tags=["deletreo", "temperatura", "ambiguo"])),

    ("temp-manta", "MANTA MAS",
     [("quiere más abrigo", ["quiero otra manta", "quiero más manta",
                             "tapame más"])],
     dict(prohibido=["gruesa", "frio"],
          tags=["deletreo", "temperatura", "grado"])),

    ("temp-destapar", "BASTA MANTA",
     [("quiere que le saquen la manta",
       ["basta de manta", "no quiero la manta", "no me tapes más"])],
     dict(prohibido=["calor", "asfixia"], polaridad="negative-stop",
          tags=["deletreo", "temperatura", "parada"])),

    ("temp-pies-frios", "PIES FRIOS",
     [("tiene los pies fríos", ["tengo los pies fríos",
                                "se me enfriaron los pies",
                                "los pies fríos"])],
     dict(prohibido=["medias", "manta"],
          tags=["deletreo", "temperatura", "sintoma"])),
]

# -------------------------------------------------------------------- personas

PERSONAS = [
    ("per-enfermera", "ENFERMERA",
     [("pide que venga la enfermera", ["llamá a la enfermera",
                                       "quiero a la enfermera",
                                       "necesito a la enfermera"])],
     dict(prohibido=["urgente", "otra"], tags=["deletreo", "personas"])),

    ("per-doctor", "DOCTOR",
     [("pide que venga el médico", ["llamá al doctor",
                                    "quiero ver al doctor",
                                    "necesito al doctor"])],
     dict(prohibido=["urgente", "guardia"], tags=["deletreo", "personas"])),

    ("per-llamar-hija", "LLAMAR HIJA",
     [("pide que llamen a su hija", ["llamá a mi hija",
                                     "quiero hablar con mi hija",
                                     "llamen a mi hija"])],
     dict(prohibido=["urgente", "telefono"],
          tags=["deletreo", "personas", "verbo-objeto"])),

    ("per-solo", "QUIERO SOLO",
     [("quiere quedarse solo", ["quiero estar solo", "quiero quedarme solo",
                                "prefiero estar solo"])],
     dict(prohibido=["un rato", "enojado", "siempre"],
          tags=["deletreo", "personas", "estado"],
          nota="El modelo tiende a suavizarlo con «un ratito», que inventa una "
               "duración que la entrada no dice.")),

    ("per-no-solo", "NO SOLO",
     [("no quiere quedarse solo", ["no quiero estar solo",
                                   "no me dejes solo", "solo no"])],
     dict(prohibido=["miedo", "un rato"], polaridad="negative",
          tags=["deletreo", "personas", "negacion"])),

    ("per-quedate", "QUEDATE",
     [("pide que se queden", ["quedate", "no te vayas", "quedate conmigo"])],
     dict(prohibido=["miedo", "un rato"], tags=["deletreo", "personas"])),

    ("per-gracias-enfermera", "GRACIAS ENFERMERA",
     [("agradece a la enfermera", ["gracias, enfermera",
                                   "gracias por todo, enfermera"])],
     dict(prohibido=["siempre"], tags=["deletreo", "cortesia", "personas"])),
]

# ---------------------------------------------------------------------- entorno

ENTORNO = [
    ("ent-luz-prender", "LUZ SI",
     [("quiere la luz prendida", ["prendé la luz", "quiero la luz prendida",
                                  "sí, la luz"])],
     dict(prohibido=["lampara", "fuerte"],
          tags=["deletreo", "entorno", "respuesta"])),

    ("ent-luz-apagar", "LUZ NO",
     [("quiere la luz apagada", ["no quiero la luz", "luz no",
                                 "no, la luz no"])],
     dict(prohibido=["molesta", "dormir"], polaridad="negative",
          tags=["deletreo", "entorno", "negacion"])),

    ("ent-ventana-abrir", "ABRIR VENTANA",
     [("pide que abran la ventana", ["abrí la ventana",
                                     "quiero la ventana abierta",
                                     "abran la ventana"])],
     dict(prohibido=["calor", "aire"],
          tags=["deletreo", "entorno", "verbo-objeto"])),

    ("ent-puerta-cerrar", "CERRAR PUERTA",
     [("pide que cierren la puerta", ["cerrá la puerta",
                                      "quiero la puerta cerrada",
                                      "cierren la puerta"])],
     dict(prohibido=["ruido", "llave", "gente"],
          tags=["deletreo", "entorno", "verbo-objeto"])),

    ("ent-ruido", "MUCHO RUIDO",
     [("hay demasiado ruido", ["hay mucho ruido", "está muy ruidoso",
                               "demasiado ruido"])],
     dict(prohibido=["television", "pasillo"],
          tags=["deletreo", "entorno", "grado"])),

    ("ent-tele-apagar", "TELEVISION NO",
     [("no quiere la televisión", ["no quiero la televisión",
                                   "televisión no", "no, la televisión no"])],
     dict(prohibido=["aburrida", "ruido"], polaridad="negative",
          tags=["deletreo", "entorno", "negacion"])),

    ("ent-musica", "MUSICA",
     [("quiere música", ["quiero música", "poné música",
                         "música, por favor"]),
      ("se refiere a la música que suena", ["la música", "esa música"])],
     dict(prohibido=["fuerte", "clasica", "radio"], ambiguo=True,
          tags=["deletreo", "entorno", "ambiguo"])),

    ("ent-volumen-menos", "VOLUMEN MENOS",
     [("quiere menos volumen", ["bajá el volumen", "quiero menos volumen",
                                "más bajo"])],
     dict(prohibido=["apaga", "molesta"],
          tags=["deletreo", "entorno", "grado"])),

    ("ent-anteojos", "ANTEOJOS",
     [("pide sus anteojos", ["quiero mis anteojos", "pasame los anteojos",
                             "necesito los anteojos"])],
     dict(prohibido=["rotos", "sucios", "leer"], tags=["deletreo", "entorno"])),
]

NUEVOS_B = DEGLUCION + TEMPERATURA + PERSONAS + ENTORNO
