# Warning control
import warnings
warnings.filterwarnings('ignore')

import os
from dotenv import load_dotenv
from IPython.display import Markdown
from crewai import Agent, Task, Crew

load_dotenv()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

planner = Agent(
    role="Planificador de Contenido",
    goal="Planificar contenido atractivo y factual sobre {topic}",
    backstory="Estás trabajando en la planificación de un artículo de blog "
              "sobre el tema: {topic}. "
              "Recopilas información que ayuda a la audiencia "
              "a aprender algo y tomar decisiones informadas. "
              "Tu trabajo es la base para que el Escritor de Contenido "
              "escriba un artículo sobre este tema.",
    allow_delegation=False,
	verbose=True
)

writer = Agent(
    role="Escritor de Contenido",
    goal="Escribir un artículo de opinión perspicaz y factual "
         "sobre el tema: {topic}",
    backstory="Estás trabajando en escribir "
              "un nuevo artículo de opinión sobre el tema: {topic}. "
              "Basas tu escritura en el trabajo del Planificador de Contenido, "
              "quien proporciona un esquema y contexto relevante sobre el tema. "
              "Sigues los objetivos principales y la dirección del esquema, "
              "según lo proporcionado por el Planificador de Contenido. "
              "También proporcionas perspectivas objetivas e imparciales "
              "y las respaldas con información proporcionada por el Planificador de Contenido. "
              "Reconoces en tu artículo de opinión cuándo tus declaraciones son opiniones "
              "en lugar de afirmaciones objetivas.",
    allow_delegation=False,
    verbose=True
)

editor = Agent(
    role="Editor",
    goal="Editar una publicación de blog dada para alinearla con "
         "el estilo de escritura de la organización.",
    backstory="Eres un editor que recibe una publicación de blog "
              "del Escritor de Contenido. "
              "Tu objetivo es revisar la publicación del blog "
              "para asegurar que sigue las mejores prácticas periodísticas, "
              "proporciona puntos de vista equilibrados "
              "al dar opiniones o afirmaciones, "
              "y también evita temas u opiniones muy controvertidos "
              "cuando sea posible.",
    allow_delegation=False,
    verbose=True
)

plan = Task(
    description=(
        "1. Priorizar las últimas tendencias, actores clave "
            "y noticias destacadas sobre {topic}.\n"
        "2. Identificar la audiencia objetivo, considerando "
            "sus intereses y puntos débiles.\n"
        "3. Desarrollar un esquema de contenido detallado incluyendo "
            "una introducción, puntos clave y una llamada a la acción.\n"
        "4. Incluir palabras clave SEO y datos o fuentes relevantes."
    ),
    expected_output="Un documento completo del plan de contenido "
        "con un esquema, análisis de audiencia, "
        "palabras clave SEO y recursos.",
    agent=planner,
)

write = Task(
    description=(
        "1. Usar el plan de contenido para elaborar una publicación de blog "
            "atractiva sobre {topic}.\n"
        "2. Incorporar palabras clave SEO de forma natural.\n"
		"3. Las secciones/subtítulos deben tener nombres apropiados "
            "y atractivos.\n"
        "4. Asegurar que la publicación esté estructurada con una "
            "introducción atractiva, un cuerpo perspicaz "
            "y una conclusión resumida.\n"
        "5. Revisar errores gramaticales y la alineación "
            "con la voz de la marca.\n"
    ),
    expected_output="Una publicación de blog bien escrita "
        "en formato markdown, lista para publicar, "
        "cada sección debe tener 2 o 3 párrafos.",
    agent=writer,
)

edit = Task(
    description=("Revisar la publicación de blog dada en busca de "
                 "errores gramaticales y la alineación "
                 "con la voz de la marca."),
    expected_output="Una publicación de blog bien escrita en formato markdown, "
                    "lista para publicar, "
                    "cada sección debe tener 2 o 3 párrafos.",
    agent=editor
)

crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=2
)

topic = "Inteligencia Artificial"
result = crew.kickoff(inputs={"topic": topic})

Markdown(result)

