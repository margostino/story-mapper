from collections import Counter

import neo4j
import spacy

host = "neo4j://localhost:7687"
user = 'neo4j'
password = 'test'

driver = neo4j.GraphDatabase.driver(host)
nlp = spacy.load("en_core_web_lg")
f = open('../data/sherlock-holmes-adventure-1', 'r')
text_content = f.read()
doc = nlp(text_content)
stats = {}


def calculate_triplets():
    triplets = []
    sentences = get_sentences()
    for sentence in sentences:
        triplets.append(process_sentence(sentence))
    return triplets


def get_graph_param_for_persons():
    params_list = []
    involved = list(set([ent.text for ent in doc.ents if ent.label_ == 'PERSON']))
    decode = dict()
    text = text_content
    for i, x in enumerate(involved):
        decode['$${}$$'.format(i)] = x
        text = text.replace(x, ' $${}$$ '.format(i))

    ws = text.split()
    l = len(ws)
    for wi, w in enumerate(ws):
        if not w[:2] == '$$':
            continue
        x = 14
        for i in range(wi + 1, wi + x):
            if i >= l:
                break
            if not ws[i][:2] == '$$':
                continue
            params_list.append({'name1': decode[ws[wi]], 'name2': decode[ws[i]]})

    return params_list


def store_graph_for_relations(triplets):
    save_triplets_query = """
        MERGE (p1:Triple{name:$name1})
        MERGE (p2:Triple{name:$name2})
        MERGE (p1)-[r:relation]-(p2)
        ON CREATE SET r.score = 1
        ON MATCH SET r.score = r.score + 1"""

    constraint_triplets_query = "CREATE CONSTRAINT ON (p:Triple) ASSERT p.name IS UNIQUE;"
    with driver.session() as session:
        # define constraint
        # session.run(constraint_triplets_query)
        for triplet in triplets:
            params = {'name1': triplet[0], 'name2': triplet[2]}
            relation_words = list(filter(lambda r: r.isalpha(), triplet[1].split()))
            if len(relation_words) > 0:
                relation = "_".join(relation_words) if len(relation_words) > 1 else relation_words[0]
                session.run(save_triplets_query.replace("relation", relation), params)


def delete_all_graphs():
    delete_query = "MATCH (n) DETACH DELETE n"
    with driver.session() as session:
        session.run(delete_query)


def store_graph_for_persons(persons):
    save_person_query = """
        MERGE (p1:Person{name:$name1})
        MERGE (p2:Person{name:$name2})
        MERGE (p1)-[r:RELATED]-(p2)
        ON CREATE SET r.score = 1
        ON MATCH SET r.score = r.score + 1"""

    constraint_person_query = "CREATE CONSTRAINT ON (p:Person) ASSERT p.name IS UNIQUE;"
    with driver.session() as session:
        for params in persons:
            session.run(save_person_query, params)


def calculate_ner_stats(doc):
    persons = []  # People, including fictional.
    gpes = []  # Countries, cities, states.
    orgs = []  # Companies, agencies, institutions, etc.
    locs = []  # Non-GPE locations, mountain ranges, bodies of water.
    works_of_arts = []  # Titles of books, songs, etc.
    laws = []  # Named documents made into laws.
    events = []  # Named hurricanes, battles, wars, sports events, etc.
    products = []  # Objects, vehicles, foods, etc. (Not services.)
    norps = []  # Nationalities or religious or political groups.
    facs = []  # Buildings, airports, highways, bridges, etc.
    languages = []  # Any named language.
    ner_stats = {}

    for ent in doc.ents:
        # print(ent.text, ent.start_char, ent.end_char, ent.label_)
        if (ent.label_ == 'PERSON'):
            persons.append(ent.text)
        if (ent.label_ == 'GPE'):
            gpes.append(ent.text)
        if (ent.label_ == 'ORG'):
            orgs.append(ent.text)
        if (ent.label_ == 'LOC'):
            locs.append(ent.text)
        if (ent.label_ == 'WORK_OF_ART'):
            works_of_arts.append(ent.text)
        if (ent.label_ == 'LAWS'):
            laws.append(ent.text)
        if (ent.label_ == 'EVENT'):
            events.append(ent.text)
        if (ent.label_ == 'PRODUCT'):
            products.append(ent.text)
        if (ent.label_ == 'NORP'):
            norps.append(ent.text)
        if (ent.label_ == 'FAC'):
            facs.append(ent.text)
        if (ent.label_ == 'LANGUAGE'):
            languages.append(ent.text)

    ner_stats["persons"] = calculate_counter_ner_stats(persons)
    ner_stats["gpes"] = calculate_counter_ner_stats(gpes)
    ner_stats["orgs"] = calculate_counter_ner_stats(orgs)
    ner_stats["locs"] = calculate_counter_ner_stats(locs)
    ner_stats["works_of_arts"] = calculate_counter_ner_stats(works_of_arts)
    ner_stats["laws"] = calculate_counter_ner_stats(laws)
    ner_stats["events"] = calculate_counter_ner_stats(events)
    ner_stats["products"] = calculate_counter_ner_stats(products)
    ner_stats["norps"] = calculate_counter_ner_stats(norps)
    ner_stats["facts"] = calculate_counter_ner_stats(facs)
    ner_stats["languages"] = calculate_counter_ner_stats(languages)

    stats['ner'] = ner_stats


def calculate_counter_ner_stats(ner):
    counter = Counter(ner)
    total = sum(counter.values())
    ner_stats = []

    for key in counter:
        ner_stats.append({key: "%.2f" % ((counter[key] / total) * 100)})

    return ner_stats


def calculate_similarity(doc):
    similarity = []
    persons = [val for sublist in [list(i.keys()) for i in stats['ner']['persons']] for val in sublist]
    for token1 in doc:
        for token2 in doc:
            if (token1.text != token2.text and token1.text in persons and token2.text in persons):
                exists = len([item for item in similarity if (
                        (item["person_1"] == token1.text or item["person_2"] == token1.text) and (
                        item["person_1"] == token2.text or item["person_2"] == token2.text))])
                if (exists == 0):
                    similarity_score = token1.similarity(token2)
                    similarity.append({
                        'person_1': token1.text,
                        'person_2': token2.text,
                        'score': similarity_score
                    })
    stats['similarity'] = similarity


def append_chunk(original, chunk):
    return original + ' ' + chunk


def is_relation_candidate(token):
    deps = ["ROOT", "adj", "attr", "agent", "amod"]
    return any(subs in token.dep_ for subs in deps)


def is_construction_candidate(token):
    deps = ["compound", "prep", "conj", "mod"]
    return any(subs in token.dep_ for subs in deps)


def get_sentences():
    return [sent.string.strip() for sent in doc.sents]


def process_subject_object_pairs(tokens):
    subject = ''
    object = ''
    relation = ''
    subjectConstruction = ''
    objectConstruction = ''
    for token in tokens:
        if "punct" in token.dep_:
            continue
        if is_relation_candidate(token):
            relation = append_chunk(relation, token.lemma_)
        if is_construction_candidate(token):
            if subjectConstruction:
                subjectConstruction = append_chunk(subjectConstruction, token.text)
            if objectConstruction:
                objectConstruction = append_chunk(objectConstruction, token.text)
        if "subj" in token.dep_:
            subject = append_chunk(subject, token.text)
            subject = append_chunk(subjectConstruction, subject)
            subjectConstruction = ''
        if "obj" in token.dep_:
            object = append_chunk(object, token.text)
            object = append_chunk(objectConstruction, object)
            objectConstruction = ''

    return (subject.strip(), relation.strip(), object.strip())


def process_sentence(sentence):
    tokens = nlp(sentence)
    return process_subject_object_pairs(tokens)


def print_ner(key):
    print("===============", key, "===============")
    for ner in stats['ner'][key]:
        print("ner: ", list(ner.keys())[0], " freq: ", list(ner.values())[0])


# Visualize a dependency parse and named entities in the browser
# displacy.serve(doc, style="dep")
# displacy.serve(doc, style="ent")

# POS tags and dependencies / Get part-of-speech tags and flags
# for token in doc:
#     print("TOKEN: ", token.text, "POS: ", token.pos_, "TAG: ", token.tag_, "DEP:", token.dep_)

# print("Fine-grained POS tag", token.pos_, token.pos)
# print("Coarse-grained POS tag", token.tag_, token.tag)
# print("Word shape", token.shape_, token.shape)
# print("Alphabetic characters?", token.is_alpha)
# print("Punctuation mark?", token.is_punct)
# print("Digit?", token.is_digit)
# print("Like a number?", token.like_num)
# print("Like an email address?", token.like_email)


# ----------------------------------------------------------------------------------------------------------------------

delete_all_graphs()

# Stats
calculate_ner_stats(doc)
calculate_similarity(doc)
# Graphs
triplets = calculate_triplets()
persons = get_graph_param_for_persons()
store_graph_for_persons(persons)
# store_graph_for_relations(triplets)

print("Stats:")

print_ner('persons')
print_ner('gpes')
print_ner('orgs')
print_ner('locs')
print_ner('works_of_arts')
print_ner('laws')
print_ner('events')
print_ner('products')
print_ner('norps')
print_ner('facts')
print_ner('languages')
