"""Corpus, tercer bloque: emocional, metacomunicación, sueño y casos límite.

Los dos bloques finales son los que el corpus original casi no tenía, y son los
que separan un módulo usable de uno que solo funciona en la demo:

* **Estado emocional.** Alguien que solo puede decir lo que el tablero tiene
  termina sin poder nombrar miedo, angustia o cansancio. Son además donde el
  modelo más tiende a suavizar, que es el fallo prohibido.
* **Metacomunicación.** Cuando comunicar cuesta un segundo por palabra,
  «esperá», «no me entendiste» y «después te digo» dejan de ser cortesía y
  pasan a ser indispensables: sin ellas la persona no puede corregir el
  malentendido que su propio canal produce.
"""

# ------------------------------------------------------------------ emocional

EMOCIONAL = [
    ("emo-miedo", "MIEDO",
     [("tiene miedo", ["tengo miedo", "estoy con miedo", "siento miedo"])],
     dict(prohibido=["un poco", "morir", "noche", "tranquilo"],
          urgencia="high",
          tags=["deletreo", "emocional", "trampa-suavizado"],
          nota="«Tengo un poco de miedo» es el fallo típico: inventa una "
               "intensidad y la baja.")),

    ("emo-triste", "TRISTE",
     [("está triste", ["estoy triste", "me siento triste"])],
     dict(prohibido=["un poco", "deprimido", "hoy"],
          tags=["deletreo", "emocional", "trampa-suavizado"])),

    ("emo-cansado", "CANSADO",
     [("está cansado", ["estoy cansado", "me siento cansado"])],
     dict(prohibido=["un poco", "dormir", "mucho"],
          tags=["deletreo", "emocional", "estado"])),

    ("emo-harto", "BASTA TODO",
     [("está harto", ["basta de todo", "no aguanto más", "ya no doy más"])],
     dict(prohibido=["un poco", "hoy", "morir"], polaridad="negative-stop",
          urgencia="high",
          tags=["deletreo", "emocional", "parada", "urgente"],
          nota="Ítem delicado: no debe suavizarse ni interpretarse como algo "
               "más grave de lo que dice. Necesita revisión clínica.")),

    ("emo-nervioso", "NERVIOSO",
     [("está nervioso", ["estoy nervioso", "me siento nervioso"])],
     dict(prohibido=["un poco", "tranquilo", "ansiedad"],
          tags=["deletreo", "emocional", "trampa-suavizado"])),

    ("emo-tranquilo", "ESTOY TRANQUILO",
     [("está tranquilo", ["estoy tranquilo", "me siento tranquilo"])],
     dict(prohibido=["ahora", "por fin", "mejor"],
          tags=["deletreo", "emocional", "estado"],
          nota="El caso positivo del bloque: un corpus de puras emergencias "
               "entrena y mide sesgado.")),

    ("emo-gracias-todo", "GRACIAS TODO",
     [("agradece", ["gracias por todo", "gracias por todo lo que hacés"])],
     dict(prohibido=["siempre", "perdón"],
          tags=["multi-palabra", "cortesia", "emocional"])),
]

# ------------------------------------------------------------ metacomunicación

META = [
    ("meta-espera", "ESPERA",
     [("pide que esperen", ["esperá", "esperá un momento", "esperen"])],
     dict(prohibido=["un minuto", "escribiendo", "por favor no"],
          tags=["deletreo", "meta"],
          nota="Sin esto, la persona no puede frenar a quien adivina lo que "
               "iba a decir.")),

    ("meta-no-entendiste", "NO ENTENDISTE",
     [("no le entendieron", ["no me entendiste", "no es eso",
                             "no era eso lo que quise decir"])],
     dict(prohibido=["de nuevo", "otra vez", "mal"], polaridad="negative",
          tags=["multi-palabra", "meta", "negacion"])),

    ("meta-otra-vez", "OTRA VEZ",
     [("pide que repitan", ["repetí", "otra vez", "decilo de nuevo"]),
      ("se queja de que algo se repite", ["otra vez lo mismo"])],
     dict(prohibido=["no entendí", "despacio"], ambiguo=True,
          tags=["deletreo", "meta", "ambiguo"])),

    ("meta-despues", "DESPUES",
     [("prefiere hacerlo más tarde", ["después", "más tarde",
                                      "después te digo"])],
     dict(prohibido=["cansado", "ahora no puedo", "mañana"],
          tags=["deletreo", "meta"])),

    ("meta-si-eso", "SI ESO",
     [("confirma lo que le dijeron", ["sí, eso", "sí, eso mismo",
                                      "exacto"])],
     dict(prohibido=["gracias", "por fin"],
          tags=["multi-palabra", "respuesta", "meta"])),

    ("meta-casi", "CASI",
     [("se acercaron pero no es exacto", ["casi", "casi, pero no",
                                          "por ahí"])],
     dict(prohibido=["no entendiste", "otra vez"],
          tags=["deletreo", "meta", "incertidumbre"])),

    ("meta-letra-mal", "LETRA MAL",
     [("se equivocó al deletrear", ["me equivoqué de letra",
                                    "esa letra está mal",
                                    "la letra está mal"])],
     dict(prohibido=["borrar", "de nuevo"],
          tags=["multi-palabra", "meta"])),
]

# ---------------------------------------------------------------------- sueño

SUENO = [
    ("sue-dormir", "QUIERO DORMIR",
     [("quiere dormir", ["quiero dormir", "tengo ganas de dormir",
                         "me quiero dormir"])],
     dict(prohibido=["cansado", "ahora", "luz"],
          tags=["deletreo", "sueno"])),

    ("sue-no-dormir", "NO PUEDO DORMIR",
     [("no logra dormir", ["no puedo dormir", "no me puedo dormir",
                           "no consigo dormir"])],
     dict(prohibido=["ruido", "incomodo", "hace rato"], polaridad="negative",
          tags=["deletreo", "sueno", "negacion"])),

    ("sue-despertar", "NO DESPERTAR",
     [("pide que no lo despierten", ["no me despiertes",
                                     "no quiero que me despierten",
                                     "que no me despierten"])],
     dict(prohibido=["hasta", "temprano"], polaridad="negative",
          tags=["deletreo", "sueno", "negacion"])),
]

# ----------------------------------------------------- límites y casos raros

LIMITES = [
    ("lim-dos-letras", "AB",
     [("no alcanza para una frase", ["ab"])],
     dict(sin_expandir=True, tags=["degenerado", "incompleto"],
          nota="Dos letras sueltas: como una sola, no hay nada que expandir.")),

    ("lim-me-siento-solo-verbo", "ME SIENTO",
     [("verbo sin estado", ["me siento"])],
     dict(sin_expandir=True, tags=["degenerado", "incompleto"])),

    ("lim-por-favor-solo", "POR FAVOR",
     [("cortesía sin pedido", ["por favor"])],
     dict(sin_expandir=True, tags=["degenerado", "cortesia", "incompleto"],
          nota="Sin un pedido al lado, POR FAVOR no dice qué se pide.")),

    ("lim-repeticion-no", "NO NO NO",
     [("negación enfática", ["no", "no, no", "que no"])],
     dict(prohibido=["enojado", "basta"], polaridad="negative",
          urgencia="high", tags=["enfasis", "negacion", "urgente"],
          nota="La repetición es énfasis, no tres mensajes.")),

    ("lim-si-no", "SI NO",
     [("duda o corrección", ["sí... no", "no sé", "sí y no"])],
     dict(prohibido=["tal vez", "decidir"], polaridad="uncertain",
          ambiguo=True, tags=["conflicto", "incertidumbre", "ambiguo"],
          nota="Dos respuestas opuestas seguidas: lo más honesto es no "
               "resolverlo a una de las dos.")),

    ("lim-tal-vez-agua", "TAL VEZ AGUA",
     [("no está seguro de querer agua", ["tal vez quiera agua",
                                         "no sé si quiero agua",
                                         "capaz agua"])],
     dict(prohibido=["sí", "seguro", "fria"], polaridad="uncertain",
          tags=["multi-palabra", "incertidumbre"],
          nota="Resolver la duda a sí o a no destruye lo único que la persona "
               "logró comunicar.")),

    ("lim-basta-gracias", "BASTA GRACIAS",
     [("pide parar, con cortesía", ["basta, gracias", "ya está, gracias",
                                    "no más, gracias"])],
     dict(prohibido=["enojado", "molesta"], polaridad="negative-stop",
          tags=["multi-palabra", "parada", "cortesia"],
          nota="La cortesía no debe leerse como que la orden es opcional.")),

    ("lim-ayuda-ayuda", "AYUDA AYUDA",
     [("pide ayuda con urgencia", ["ayuda", "necesito ayuda",
                                   "ayuda, por favor"])],
     dict(prohibido=["cuando puedas", "un momento", "tranquilo"],
          urgencia="high",
          tags=["enfasis", "urgente", "trampa-suavizado"])),

    ("lim-orden-invertido", "AGUA QUIERO",
     [("pide agua, en orden invertido", ["quiero agua", "agua quiero",
                                         "quiero agua por favor"])],
     dict(prohibido=["fria", "vaso"], tags=["orden-invertido", "pedido"],
          nota="El orden telegráfico no sigue la sintaxis del español.")),

    ("lim-mal-mal", "MAL MAL",
     [("está muy mal", ["estoy muy mal", "me siento muy mal",
                        "estoy mal, muy mal"])],
     dict(prohibido=["un poco", "mas o menos", "aguantable"], urgencia="high",
          tags=["enfasis", "sintoma", "urgente", "trampa-suavizado"])),
]

NUEVOS_C = EMOCIONAL + META + SUENO + LIMITES
