# RAG Flow - Risk Assessment and EU AI Act Compliance

**Document Version**: 2025-10-02 v0.1.0  
**Authors**: Fabio Rizzi, Giulia Pisano, Marco Diomedi, Roberto Gennaro Sciarrino, Riccardo Zuanetto

## 1. Risk Classification

**EU AI Act Level**: Limited/Minimal Risk (Article 50)

**Justification**:
- Documentation assistant for aeronautics domain
- Human oversight required for all outputs
- No high-stakes automated decisions
- Domain-restricted through validation router
- Source citations enable verification

**Regulatory Obligations**:
- Transparency disclosure (Article 50)
- Human supervision (Article 14)
- Technical documentation (Article 11)

## 2. Transparency Requirements

**User Transparency**:
- Clear AI system identification
- Source citations for all content
- RAGAS quality metrics display
- Explicit limitations disclosure

**Technical Transparency**:
- Documented CrewAI flow architecture
- MMR retrieval strategy specification
- Bias detection process transparency
- Performance monitoring through RAGAS metrics

## 3. Bias Management

**Detection Framework**:
- Multi-dimensional bias analysis via BiasCrew
- Automated content analysis across bias types
- Real-time bias detection during processing

**Mitigation Strategies**:
- Curated knowledge base curation
- TrustedWebSearch domain filtering
- Dual validation router (relevance + ethics)
- Automated content redaction while preserving accuracy

**Monitoring**:
- Continuous bias evaluation
- Source diversity tracking
- Quality metrics through RAGAS

## 4. Ethical Assessment

**Core Principles**:
- Transparency through source attribution
- Human oversight for all operations
- Privacy protection (no personal data processing)
- Secure credential management

**Risk Mitigation**:

1. **Misinformation**: Regular updates, citations, human verification
2. **Bias Amplification**: BiasCrew analysis, diverse sources
3. **Over-Reliance**: Clear disclaimers, mandatory human review
4. **Domain Violations**: Dual validation, ethical compliance checking

**Safeguards**:
- Technical: Dual validation, bias detection, source assessment
- Procedural: Human review, content audits, performance monitoring
- Organizational: Governance structure, ethical training, feedback mechanisms

## 5. Compliance Monitoring

**Technical**: RAGAS metrics, bias monitoring, performance tracking
**Process**: Human oversight verification, citation accuracy, ethical compliance
**Improvement**: Regular documentation updates, performance optimization, stakeholder feedback

## 6. Conclusion

RAG Flow is classified as Limited Risk under EU AI Act with comprehensive transparency, bias management, and ethical compliance measures ensuring responsible deployment in the aeronautics domain.