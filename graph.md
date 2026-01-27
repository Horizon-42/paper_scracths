graph TD
    %% 定义样式
    classDef input fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1,rx:10,ry:10;
    classDef process fill:#FFFFFF,stroke:#37474F,stroke-width:2px,color:#000000,rx:5,ry:5;
    classDef core_innovation fill:#FFF8E1,stroke:#FF8F00,stroke-width:3px,color:#000000,rx:5,ry:5,stroke-dasharray: 5 5;
    classDef db fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20,shape:cyl;
    classDef output fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#4A148C,rx:10,ry:10;
    classDef invisible fill:none,stroke:none;

    %% --- Layer 1: Data Collection ---
    subgraph L1 ["Layer 1: Data Collection & Initial Decoding"]
        style L1 fill:#eceff1,stroke:#cfd8dc
        Input_Route([Flight Route Query]):::input --> Data_Coll[Multisource Data Collection];
        Data_Coll -- "Raw Data (METAR/TAF, etc.)" --> Decoder[Rule-based Decoder];
    end

    %% 外部数据库连接
    DB_Sources[(Public DBs<br/>NTSB, NWS-AFD)]:::db -.-> Data_Coll;

    Decoder -- "Decoded Language Chunks" --> GRAG_Layer;

    %% --- Layer 2: Graph RAG Enhance (核心创新) ---
    subgraph L2 ["Layer 2: Graph RAG Enhancement Layer (Core Innovation)"]
        style L2 fill:#fffde7,stroke:#ffe082
        GRAG_Layer{{Graph RAG Engine}}:::core_innovation;
        
        KG[(Aviation Knowledge Graph DB<br/>3-Layer Community Structure)]:::db <==>|Beam-Search Retrieval| GRAG_Layer;
        
        %% RAG的具体任务
        GRAG_Layer -.-> Task1(Weather Classification);
        GRAG_Layer -.-> Task2(Operational Suggestions);
        GRAG_Layer -.-> Task3(Go/No-Go Risk Assessment);
        
        Task1 & Task2 & Task3 -.->|Structured Intermediate Report| GRAG_Output[Enhanced Structured Context];
    end

    GRAG_Output --> Numeric_Layer;

    %% --- Layer 3: Numeric Logic (核心创新) ---
    subgraph L3 ["Layer 3: Numeric Logic Preprocessing (Core Innovation)"]
        style L3 fill:#fffde7,stroke:#ffe082
        Numeric_Layer{{Numeric Logic Processor}}:::core_innovation;
        Note_Num[/"Adopt NumeroLogic concept to<br/>enhance numerical understanding"/]:::invisible --- Numeric_Layer;
    end
    
    Numeric_Layer -- "Numerically Enhanced Prompt" --> Gen_Layer;

    %% --- Layer 4: Narrative Generation ---
    subgraph L4 ["Layer 4: Narrative Generation Layer"]
        style L4 fill:#eceff1,stroke:#cfd8dc
        Gen_Layer[Fine-tuned LLM<br/>(Teacher-Student / RALA-DPO Pipeline)]:::process;
        
        Train_DB[(Critical Scenario<br/>Training DB)]:::db -.->|Fine-tuning Pipeline| Gen_Layer;
    end

    Gen_Layer --> Final_Output([Flight Route Weather Briefing]):::output;

    %% 连接线注释
    linkStyle 2,5,11,14,16 stroke-width:2px,fill:none,stroke:black;