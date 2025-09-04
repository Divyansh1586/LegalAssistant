from neo4j import GraphDatabase
import json

# Connection to Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "abcdefgh"))

# Load JSON (list of fragments)
with open("graph_fragments.json") as f:
    fragments = json.load(f)

def load_fragment(tx, fragment):
    # Create nodes
    for node in fragment.get("nodes", []):
        query = f"""
        MERGE (n:{node['label']} {{id: $id}})
        SET n += $props
        """
        tx.run(query, id=node["id"], props=node.get("properties", {}))

    # Create edges
    for edge in fragment.get("edges", []):
        query = f"""
        MATCH (a {{id: $src}}), (b {{id: $tgt}})
        MERGE (a)-[r:{edge['type']}]->(b)
        SET r += $props
        """
        tx.run(
            query,
            src=edge["source_id"],
            tgt=edge["target_id"],
            props=edge.get("properties", {})
        )


# Write fragments into Neo4j
with driver.session() as session:
    for fragment in fragments:
        session.execute_write(load_fragment, fragment)

driver.close()
