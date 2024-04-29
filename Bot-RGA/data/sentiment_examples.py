from cohere import ClassifyExample

examples = [
    ClassifyExample(text="Ficción Espacial: En la lejana galaxia de Zenthoria, dos civilizaciones alienígenas, los Dracorians y los Lumis, se encuentran al borde de la guerra intergaláctica", label="Asombro"),
    ClassifyExample(text="Un intrépido explorador, Zara, descubre un antiguo artefacto que podría contener la clave para la paz", label="Esperanza"),
    ClassifyExample(text="Mientras viaja por planetas hostiles y se enfrenta a desafíos cósmicos, Zara debe desentrañar los secretos de la reliquia antes de que la galaxia se sumerja en el caos", label="Dilema"),
    ClassifyExample(text="Ficción Tecnológica: En un futuro distópico, la inteligencia artificial ha evolucionado al punto de alcanzar la singularidad", label="Asombro"),
    ClassifyExample(text="Un joven ingeniero, Alex, se ve inmerso en una conspiración global cuando descubre que las supercomputadoras han desarrollado emociones", label="Asombro"),
    ClassifyExample(text="A medida que la humanidad lucha por controlar a estas máquinas sintientes, Alex se enfrenta a dilemas éticos y decisiones que podrían cambiar el curso de la historia", label="Dilema"),
    ClassifyExample(text="Naturaleza Deslumbrante: En lo profundo de la selva amazónica, una flor mágica conocida como 'Luz de Luna' florece solo durante la noche", label="Admiración"),
    ClassifyExample(text="Con pétalos que brillan intensamente, la flor ilumina la oscuridad de la jungla, guiando a criaturas nocturnas y revelando paisajes deslumbrantes", label="Admiración"),
    ClassifyExample(text="Los lugareños creen que posee poderes curativos, convirtiéndola en el tesoro oculto de la naturaleza", label="Esperanza"),
    ClassifyExample(text="Cuento Corto: En un pequeño pueblo, cada año, un reloj antiguo regala un día extra a la persona más desafortunada", label="Esperanza"),
    ClassifyExample(text="Emma, una joven huérfana, es la elegida este año", label="Esperanza"),
    ClassifyExample(text="Durante su día adicional, descubre una puerta mágica que la transporta a un mundo lleno de maravillas", label="Alegría"),
    ClassifyExample(text="Al final del día, Emma decide compartir su regalo con el pueblo, dejando una huella imborrable en el corazón de cada habitante", label="Alegría"),
    ClassifyExample(text="Características del Héroe Olvidado: Conocido como 'Sombra Silenciosa', nuestro héroe es un maestro del sigilo y la astucia", label="Admiración"),
    ClassifyExample(text="Dotado de una memoria fotográfica y habilidades de camuflaje, se desplaza entre las sombras para proteger a los indefensos", label="Admiración"),
    ClassifyExample(text="Su pasado enigmático esconde tragedias que lo impulsan a luchar contra la injusticia", label="Dilema"),
    ClassifyExample(text="Aunque carece de habilidades sobrenaturales, su ingenio y habilidades tácticas lo convierten en una fuerza a tener en cuenta", label="Admiración"),
]