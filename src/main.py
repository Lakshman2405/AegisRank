# src/main.py
import os
import sys
import json
from typing import List
from agents.ingestion_agent import initialize_graph_runtime, ConcurrentStreamState

class IngestionStreamCoordinator:
    """Manages raw candidate structures and handles stream data serialization."""
    def __init__(self, data_source_path: str, output_sink_path: str):
        self.data_source_path = data_source_path
        self.output_sink_path = output_sink_path
        self.processing_graph = initialize_graph_runtime()
        
        # Initialize the output file stream cleanly
        os.makedirs(os.path.dirname(self.output_sink_path), exist_ok=True)
        with open(self.output_sink_path, "w", encoding="utf-8") as file_handle:
            pass 

    def execute_stream_ingestion(self, batch_allocation_limit: int = 5000):
        """Processes line-by-line streaming to manage memory footprint efficiently."""
        if not os.path.exists(self.data_source_path):
            print(f"Critical Error: Source log dataset not found at {self.data_source_path}")
            sys.exit(1)
            
        print(f"Initializing data stream ingestion from: {self.data_source_path}")
        print(f"Targeting append-only serialization file at: {self.output_sink_path}")
        print("\n=== STARTING MULTI-AGENT INGESTION COGNITIVE FILTERS ===")
        
        current_buffer = []
        global_offset = 0
        total_serialized_records = 0
        total_anomalies_purged = 0
        
        with open(self.data_source_path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                if not line.strip():
                    continue
                current_buffer.append(line)
                
                if len(current_buffer) == batch_allocation_limit:
                    saved, anomalies = self._process_buffer_chunk(current_buffer, global_offset)
                    global_offset += len(current_buffer)
                    total_serialized_records += saved
                    total_anomalies_purged += anomalies
                    current_buffer = []
            
            # Flush remaining buffer lines
            if current_buffer:
                saved, anomalies = self._process_buffer_chunk(current_buffer, global_offset)
                global_offset += len(current_buffer)
                total_serialized_records += saved
                total_anomalies_purged += anomalies
                
        print("\n=== MULTI-AGENT INGESTION PIPELINE CONCLUDED SUCCESSFULLY ===")
        print(f" Total Ingested Line Signals:  {global_offset}")
        print(f" Total Coherent Records Saved: {total_serialized_records} -> Stored at {self.output_sink_path}")
        print(f" Total Cognitive Traps Purged: {total_anomalies_purged}")

    def _process_buffer_chunk(self, buffer_chunk: List[str], offset: int) -> tuple:
        print(f"Streaming block slice: indices [{offset} to {offset + len(buffer_chunk)}]")
        
        initial_runtime_state = ConcurrentStreamState(
            raw_buffer_chunk=buffer_chunk,
            validated_structural_records=[],
            audited_production_pool=[],
            anomaly_telemetry_counter=0
        )
        
        execution_output = self.processing_graph.invoke(initial_runtime_state)
        
        # Stream valid records directly to the file on disk
        saved_count = 0
        with open(self.output_sink_path, "a", encoding="utf-8") as out_file:
            for profile in execution_output["audited_production_pool"]:
                record_payload = {
                    "candidate_id": profile.candidate_id,
                    "semantic_narrative_block": profile.semantic_narrative_block,
                    "telemetry": profile.telemetry.model_dump()
                }
                out_file.write(json.dumps(record_payload) + "\n")
                saved_count += 1
        
        print(f" ├── Schema verification passed: {len(execution_output['validated_structural_records'])} rows")
        print(f" ├── Safely written to local disk: {saved_count} profiles")
        print(f" └── AI Fraud Traps Removed:    {execution_output['anomaly_telemetry_counter']} records")
        
        return saved_count, execution_output["anomaly_telemetry_counter"]

if __name__ == "__main__":
    production_dataset = os.path.join("data", "candidates.jsonl")
    clean_output_sink = os.path.join("data", "clean_candidates.jsonl")
    
    coordinator = IngestionStreamCoordinator(
        data_source_path=production_dataset,
        output_sink_path=clean_output_sink
    )
    coordinator.execute_stream_ingestion(batch_allocation_limit=5000)