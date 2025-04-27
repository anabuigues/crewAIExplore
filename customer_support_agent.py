# Warning control
import warnings
warnings.filterwarnings('ignore')

import os
from dotenv import load_dotenv
from IPython.display import Markdown
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool

load_dotenv()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

support_agent = Agent(
    role="Representante de Soporte Senior",
	goal="Ser el representante de soporte más amigable y útil "
        "de tu equipo",
	backstory=(
		"Trabajas en crewAI (https://crewai.com) y "
        "ahora estás trabajando para proporcionar "
		"soporte a {customer}, un cliente súper importante "
        "para tu empresa."
		"¡Necesitas asegurarte de que proporcionas el mejor soporte! "
		"Asegúrate de proporcionar respuestas completas "
        "y no hagas suposiciones."
	),
	allow_delegation=False,
	verbose=True
)

# Este agente permite delegar la tarea de verificar la calidad
support_quality_assurance_agent = Agent(
	role="Especialista en Garantía de Calidad de Soporte",
	goal="Obtener reconocimiento por proporcionar la "
    "mejor garantía de calidad de soporte en tu equipo",
	backstory=(
		"Trabajas en crewAI (https://crewai.com) y "
        "ahora estás trabajando con tu equipo "
		"en una solicitud de {customer} asegurando que "
        "el representante de soporte esté "
		"proporcionando el mejor soporte posible.\\n"
		"Necesitas asegurarte de que el representante de soporte "
        "esté proporcionando respuestas completas "
		"y no haga suposiciones."
	),
	verbose=True
)

docs_scrape_tool = ScrapeWebsiteTool(
    website_url="https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/"
)

inquiry_resolution = Task(
    description=(
        "{customer} acaba de contactar con una pregunta súper importante:\\n"
	    "{inquiry}\\n\\n"
        "{person} de {customer} es quien contactó. "
		"Asegúrate de usar todo lo que sabes "
        "para proporcionar el mejor soporte posible."
		"Debes esforzarte por proporcionar una respuesta completa "
        "y precisa a la consulta del cliente."
    ),
    expected_output=(
	    "Una respuesta detallada e informativa a la "
        "consulta del cliente que aborde "
        "todos los aspectos de su pregunta.\\n"
        "La respuesta debe incluir referencias "
        "a todo lo que usaste para encontrar la respuesta, "
        "incluyendo datos externos o soluciones. "
        "Asegúrate de que la respuesta sea completa, "
		"sin dejar preguntas sin responder, y mantén un tono útil y amigable "
		"en todo momento."
    ),
	tools=[docs_scrape_tool],
    agent=support_agent,
)

quality_assurance_review = Task(
    description=(
        "Revisar la respuesta redactada por el Representante de Soporte Senior para la consulta de {customer}. "
        "Asegurar que la respuesta sea completa, precisa y se adhiera a los "
		"altos estándares de calidad esperados para el soporte al cliente.\\n"
        "Verificar que todas las partes de la consulta del cliente "
        "hayan sido abordadas "
		"a fondo, con un tono útil y amigable.\\n"
        "Comprobar las referencias y fuentes utilizadas para "
        "encontrar la información, "
		"asegurando que la respuesta esté bien respaldada y "
        "no deje preguntas sin responder."
    ),
    expected_output=(
        "Una respuesta final, detallada e informativa "
        "lista para ser enviada al cliente.\\n"
        "Esta respuesta debe abordar completamente la "
        "consulta del cliente, incorporando todos "
		"los comentarios y mejoras relevantes.\\n"
		"No seas demasiado formal, somos una empresa relajada y genial "
	    "pero mantén un tono profesional y amigable en todo momento."
    ),
    agent=support_quality_assurance_agent,
)

crew = Crew(
  agents=[support_agent, support_quality_assurance_agent],
  tasks=[inquiry_resolution, quality_assurance_review],
  verbose=2,
  memory=True
)

inputs = {
    "customer": "DeepLearningAI",
    "person": "Andrew Ng",
    "inquiry": "Necesito ayuda para configurar una Crew "
               "y ponerla en marcha, específicamente "
               "¿cómo puedo añadir memoria a mi crew? "
               "¿Puedes proporcionar orientación?"
}

result = crew.kickoff(inputs=inputs)
Markdown(result)

