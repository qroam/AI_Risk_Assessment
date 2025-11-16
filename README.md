# AI_Risk_Assessment

Implementation code for the paper *Managing Generative AI Risks Across the Data Lifecycle: A Multi-Stage Framework for Dynamic Risk Assessment*.

## Usage
1. Firstly, visit [AIID](https://incidentdatabase.ai/apps/incidents/) and [Anliji(人工智能风险与治理案例库)](https://www.ai-governance.online/cases-cn) to obtain webpages containing AI risk information.
2. Use `parse_AIID_webpage.py` and `parse_Anliji_webpage.py` to parst the HTMLs and convert them into json format, respectively
3. Run `proto_classification.py` to conduct prototype classification.
4. Run to `hierarchical_cluster.py` to conduct document clustering.
