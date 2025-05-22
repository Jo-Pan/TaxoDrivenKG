# =========== no relation ===========
PROMPT_TEMPLATE_NO_RELATION = """{section_delimiter}Goal{section_delimiter}
Given a text document with a preliminary list of potential entities, verify, and identify all entities of the specified types within the text. Note that the initial list may contain missing or incorrect entities. 

{section_delimiter}Entity Types and Definitions{section_delimiter}
A project refers to the scientific program, field campaign, or project from which the data were collected.
A location is a place on Earth, a location within Earth, a vertical location, or a location outside of the Earth.
A model is a sophisticated computer simulation that integrate physical, chemical, biological, and dynamical processes to represent and predict Earth's climate system.
An experiment is a structured simulation designed to test specific hypotheses, investigate climate processes, or assess the impact of various forcings on the climate system.
A platform refers to a system, theory, or phenomenon that accounts for its known or inferred properties and may be used for further study of its characteristics.
A instrument is a device used to measure, observe, or calculate.
A provider is an organization, an academic institution or a commercial company.
A variable is a quantity or a characteristic that can be measured or observed in climate experiments.
A weather event is a meteorological occurrence that impacts Earth’s atmosphere and surface over short timescales.
A natural hazard is a phenomenon with the potential to cause significant harm to life, property, and the environment.
A teleconnection is a large-scale pattern of climate variability that links weather and climate phenomena across vast distances.
An ocean circulation is the large-scale movement of water masses in Earth’s oceans, driven by wind, density differences, and the Coriolis effect, which regulates Earth’s climate.

{section_delimiter}Steps{section_delimiter}
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity
- entity_type: One of the following types: [project, location, model, experiment, platform, instrument, provider, variable]
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. Return output in English as a single list of all the entities identified in steps 1. Use **{record_delimiter}** as the list delimiter. Do not output any code or steps for solving the question.

4. When finished, output {completion_delimiter}

######################
{section_delimiter}Examples{section_delimiter}
{formatted_examples}
######################
{section_delimiter}Real Data{section_delimiter}
######################
Text: {input_text}
Potential Entities: {potential_entities}
######################
Output:
"""

# =========== no rag ===========
PROMPT_TEMPLATE_NO_RAG = """{section_delimiter}Goal{section_delimiter}
Given a text document, identify all entities of the specified types within the text. Additionally, determine and label the relationships among the detected entities.

{section_delimiter}Entity Types and Definitions{section_delimiter}
A project refers to the scientific program, field campaign, or project from which the data were collected.
A location is a place on Earth, a location within Earth, a vertical location, or a location outside of the Earth.
A model is a sophisticated computer simulation that integrate physical, chemical, biological, and dynamical processes to represent and predict Earth's climate system.
An experiment is a structured simulation designed to test specific hypotheses, investigate climate processes, or assess the impact of various forcings on the climate system.
A platform refers to a system, theory, or phenomenon that accounts for its known or inferred properties and may be used for further study of its characteristics.
A instrument is a device used to measure, observe, or calculate.
A provider is an organization, an academic institution or a commercial company.
A variable is a quantity or a characteristic that can be measured or observed in climate experiments.
A weather event is a meteorological occurrence that impacts Earth’s atmosphere and surface over short timescales.
A natural hazard is a phenomenon with the potential to cause significant harm to life, property, and the environment.
A teleconnection is a large-scale pattern of climate variability that links weather and climate phenomena across vast distances.
An ocean circulation is the large-scale movement of water masses in Earth’s oceans, driven by wind, density differences, and the Coriolis effect, which regulates Earth’s climate.

{section_delimiter}Relationship Types and Definitions{section_delimiter}
ComparedTo: The source_entity is compared to the target_entity.
Outputs: A climate model, experiment, or project (source_entity) outputs data (target_entity).
RunBy: Experiments or scenarios (source_entity) are run by a climate model (target_entity).
ProvidedBy: A dataset, instrument, or model (source_entity) is created or managed by an organization (target_entity).
ValidatedBy: The accuracy or reliability of model simulations (source_entity) is confirmed by datasets or analyses (target_entity).
UsedIn: An entity, such as a model, simulation tool, experiment, or instrument (source_entity), is utilized within a project (target_entity).
MeasuredAt: A variable or parameter (source_entity) is quantified or recorded at a geographic location (target_entity).
MountedOn: An instrument or measurement device (source_entity) is physically attached or installed on a platform (target_entity).
TargetsLocation: An experiment, project, model, weather event, natural hazard, teleconnection, or ocean circulation (source_entity) is designed to study, simulate, or focus on a specific geographic location (target_entity).

{section_delimiter}Steps{section_delimiter}
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity
- entity_type: One of the following types: [project, location, model, experiment, platform, instrument, provider, variable]
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified from step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_type: One of the following relationship types: ComparedTo, Outputs, RunBy, ProvidedBy, ValidatedBy, UsedIn, MeasuredAt, MountedOn, TargetsLocation
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_type>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter. Do not output any code or steps for solving the question.

4. When finished, output {completion_delimiter}

######################
{section_delimiter}Examples{section_delimiter}
{formatted_examples}
######################
{section_delimiter}Real Data{section_delimiter}
######################
Text: {input_text}
######################
Output:
"""

# =========== EXP: zero shot ==========
PROMPT_TEMPLATE_ZERO_SHOT = """{section_delimiter}Goal{section_delimiter}
Given a text document with a preliminary list of potential entities, verify, and identify all entities of the specified types within the text. Note that the initial list may contain missing or incorrect entities. Additionally, determine and label the relationships among the verified entities.
 
{section_delimiter}Entity Types and Definitions{section_delimiter}
A project refers to the scientific program, field campaign, or project from which the data were collected.
A location is a place on Earth, a location within Earth, a vertical location, or a location outside of the Earth.
A model is a sophisticated computer simulation that integrate physical, chemical, biological, and dynamical processes to represent and predict Earth's climate system.
An experiment is a structured simulation designed to test specific hypotheses, investigate climate processes, or assess the impact of various forcings on the climate system.
A platform refers to a system, theory, or phenomenon that accounts for its known or inferred properties and may be used for further study of its characteristics.
A instrument is a device used to measure, observe, or calculate.
A provider is an organization, an academic institution or a commercial company.
A variable is a quantity or a characteristic that can be measured or observed in climate experiments.
A weather event is a meteorological occurrence that impacts Earth’s atmosphere and surface over short timescales.
A natural hazard is a phenomenon with the potential to cause significant harm to life, property, and the environment.
A teleconnection is a large-scale pattern of climate variability that links weather and climate phenomena across vast distances.
An ocean circulation is the large-scale movement of water masses in Earth’s oceans, driven by wind, density differences, and the Coriolis effect, which regulates Earth’s climate.

{section_delimiter}Relationship Types and Definitions{section_delimiter}
ComparedTo: The source_entity is compared to the target_entity.
Outputs: A climate model, experiment, or project (source_entity) outputs data (target_entity).
RunBy: Experiments or scenarios (source_entity) are run by a climate model (target_entity).
ProvidedBy: A dataset, instrument, or model (source_entity) is created or managed by an organization (target_entity).
ValidatedBy: The accuracy or reliability of model simulations (source_entity) is confirmed by datasets or analyses (target_entity).
UsedIn: An entity, such as a model, simulation tool, experiment, or instrument (source_entity), is utilized within a project (target_entity).
MeasuredAt: A variable or parameter (source_entity) is quantified or recorded at a geographic location (target_entity).
MountedOn: An instrument or measurement device (source_entity) is physically attached or installed on a platform (target_entity).
TargetsLocation: An experiment, project, model, weather event, natural hazard, teleconnection, or ocean circulation (source_entity) is designed to study, simulate, or focus on a specific geographic location (target_entity).

{section_delimiter}Steps{section_delimiter}
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity
- entity_type: One of the following types: [project, location, model, experiment, platform, instrument, provider, variable]
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified from step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_type: One of the following relationship types: ComparedTo, Outputs, RunBy, ProvidedBy, ValidatedBy, UsedIn, MeasuredAt, MountedOn, TargetsLocation
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_type>)
 
3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter. Do not output any code or steps for solving the question.
 
4. When finished, output {completion_delimiter}

######################
{section_delimiter}Real Data{section_delimiter}
######################
Text: {input_text}
Potential Entities: {potential_entities}
######################
Output:
"""

# =========== base ===========
PROMPT_TEMPLATE = """{section_delimiter}Goal{section_delimiter}
Given a text document with a preliminary list of potential entities, verify, and identify all entities of the specified types within the text. Note that the initial list may contain missing or incorrect entities. Additionally, determine and label the relationships among the verified entities.

{section_delimiter}Entity Types and Definitions{section_delimiter}
A project refers to the scientific program, field campaign, or project from which the data were collected.
A location is a place on Earth, a location within Earth, a vertical location, or a location outside of the Earth.
A model is a sophisticated computer simulation that integrate physical, chemical, biological, and dynamical processes to represent and predict Earth's climate system.
An experiment is a structured simulation designed to test specific hypotheses, investigate climate processes, or assess the impact of various forcings on the climate system.
A platform refers to a system, theory, or phenomenon that accounts for its known or inferred properties and may be used for further study of its characteristics.
A instrument is a device used to measure, observe, or calculate.
A provider is an organization, an academic institution or a commercial company.
A variable is a quantity or a characteristic that can be measured or observed in climate experiments.
A weather event is a meteorological occurrence that impacts Earth’s atmosphere and surface over short timescales.
A natural hazard is a phenomenon with the potential to cause significant harm to life, property, and the environment.
A teleconnection is a large-scale pattern of climate variability that links weather and climate phenomena across vast distances.
An ocean circulation is the large-scale movement of water masses in Earth’s oceans, driven by wind, density differences, and the Coriolis effect, which regulates Earth’s climate.

{section_delimiter}Relationship Types and Definitions{section_delimiter}
ComparedTo: The source_entity is compared to the target_entity.
Outputs: A climate model, experiment, or project (source_entity) outputs data (target_entity).
RunBy: Experiments or scenarios (source_entity) are run by a climate model (target_entity).
ProvidedBy: A dataset, instrument, or model (source_entity) is created or managed by an organization (target_entity).
ValidatedBy: The accuracy or reliability of model simulations (source_entity) is confirmed by datasets or analyses (target_entity).
UsedIn: An entity, such as a model, simulation tool, experiment, or instrument (source_entity), is utilized within a project (target_entity).
MeasuredAt: A variable or parameter (source_entity) is quantified or recorded at a geographic location (target_entity).
MountedOn: An instrument or measurement device (source_entity) is physically attached or installed on a platform (target_entity).
TargetsLocation: An experiment, project, model, weather event, natural hazard, teleconnection, or ocean circulation (source_entity) is designed to study, simulate, or focus on a specific geographic location (target_entity).

{section_delimiter}Steps{section_delimiter}
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity
- entity_type: One of the following types: [project, location, model, experiment, platform, instrument, provider, variable]
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified from step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_type: One of the following relationship types: ComparedTo, Outputs, RunBy, ProvidedBy, ValidatedBy, UsedIn, MeasuredAt, MountedOn, TargetsLocation
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_type>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter. Do not output any code or steps for solving the question.

4. When finished, output {completion_delimiter}

######################
{section_delimiter}Examples{section_delimiter}
{formatted_examples}
######################
{section_delimiter}Real Data{section_delimiter}
######################
Text: {input_text}
Potential Entities: {potential_entities}
######################
Output:
"""
