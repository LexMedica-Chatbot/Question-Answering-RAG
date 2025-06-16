# Comprehensive RAG Benchmark Report

**Generated:** 2025-06-16T19:53:19.666967  
**Timestamp:** 20250616_195319

## Executive Summary

- **Total Tests:** 48
- **Successful Tests:** 48
- **Overall Success Rate:** 100.0%

## Test Configuration

### APIs Tested
1. **Simple API** (Port 8081) - Basic RAG pipeline
2. **Multi-Step API** (Port 8080) - Enhanced Multi-Step RAG with agent-based approach

### Embedding Models
1. **Small Embedding** - Faster, less comprehensive
2. **Large Embedding** - Slower, more comprehensive

### Test Categories
- **Definisi** - Basic definitions (Easy)
- **Perizinan** - Licensing procedures (Easy-Medium)
- **Sanksi** - Legal sanctions (Medium)
- **Prosedur** - Administrative procedures (Medium)
- **Etika** - Medical ethics (Medium)
- **Analisis** - Complex analysis (Hard)
- **Komprehensif** - Comprehensive analysis (Very Hard)

## Performance Results


### SIMPLE API + SMALL Embedding

**Overall Performance:**
- Success Rate: 100.0%
- Average Response Time: 13787 ms
- Median Response Time: 13644 ms
- Response Time Range: 4761 - 26669 ms

**Answer Quality:**
- Average Answer Length: 2020 characters
- Average Word Count: 269.7 words
- Keyword Matching Score: 0.917
- Structure Score: 0.817
- Completeness Score: 0.948
- **Overall Quality Score: 0.894**

**Technical Metrics:**
- Average Documents Retrieved: 4.0
- Average Processing Steps: 1.0

**Performance by Difficulty:**

- **Easy**: 100.0% success, 7252ms avg, 0.798 quality
- **Medium**: 100.0% success, 12700ms avg, 0.833 quality
- **Hard**: 100.0% success, 15836ms avg, 1.000 quality
- **Very_Hard**: 100.0% success, 22690ms avg, 1.000 quality


### SIMPLE API + LARGE Embedding

**Overall Performance:**
- Success Rate: 100.0%
- Average Response Time: 15845 ms
- Median Response Time: 16840 ms
- Response Time Range: 6746 - 24691 ms

**Answer Quality:**
- Average Answer Length: 2227 characters
- Average Word Count: 303.9 words
- Keyword Matching Score: 0.944
- Structure Score: 0.833
- Completeness Score: 0.962
- **Overall Quality Score: 0.913**

**Technical Metrics:**
- Average Documents Retrieved: 4.0
- Average Processing Steps: 1.0

**Performance by Difficulty:**

- **Easy**: 100.0% success, 7903ms avg, 0.798 quality
- **Medium**: 100.0% success, 17097ms avg, 0.892 quality
- **Hard**: 100.0% success, 17191ms avg, 1.000 quality
- **Very_Hard**: 100.0% success, 23238ms avg, 1.000 quality


### MULTI API + SMALL Embedding

**Overall Performance:**
- Success Rate: 100.0%
- Average Response Time: 69034 ms
- Median Response Time: 68805 ms
- Response Time Range: 21205 - 112816 ms

**Answer Quality:**
- Average Answer Length: 928 characters
- Average Word Count: 125.6 words
- Keyword Matching Score: 0.500
- Structure Score: 0.400
- Completeness Score: 0.541
- **Overall Quality Score: 0.480**

**Technical Metrics:**
- Average Documents Retrieved: 6.6
- Average Processing Steps: 4.4

**Performance by Difficulty:**

- **Easy**: 100.0% success, 30683ms avg, 0.483 quality
- **Medium**: 100.0% success, 71866ms avg, 0.833 quality
- **Hard**: 100.0% success, 86141ms avg, 0.319 quality
- **Very_Hard**: 100.0% success, 95234ms avg, 0.012 quality


### MULTI API + LARGE Embedding

**Overall Performance:**
- Success Rate: 100.0%
- Average Response Time: 68767 ms
- Median Response Time: 71038 ms
- Response Time Range: 22740 - 110316 ms

**Answer Quality:**
- Average Answer Length: 940 characters
- Average Word Count: 125.5 words
- Keyword Matching Score: 0.528
- Structure Score: 0.333
- Completeness Score: 0.555
- **Overall Quality Score: 0.472**

**Technical Metrics:**
- Average Documents Retrieved: 6.4
- Average Processing Steps: 4.5

**Performance by Difficulty:**

- **Easy**: 100.0% success, 52273ms avg, 0.724 quality
- **Medium**: 100.0% success, 72715ms avg, 0.670 quality
- **Hard**: 100.0% success, 78999ms avg, 0.264 quality
- **Very_Hard**: 100.0% success, 70264ms avg, 0.012 quality


## Comparative Analysis

### Key Findings

**🏆 Best Overall Performance:** Simple + Large
- Combined score (success rate × quality): 91.34

**⚡ Fastest Response:** Simple + Small
- Average response time: 13787 ms

**🎯 Highest Quality:** Simple + Large
- Average quality score: 0.913

**📚 Most Comprehensive:** Multi + Small
- Average documents retrieved: 6.6

## Recommendations

### For Production Use:
1. **High-Performance Requirements:** Use Simple + Small
2. **High-Quality Requirements:** Use Simple + Large  
3. **Balanced Requirements:** Use Simple + Large

### Trade-offs:
- **Simple API** offers faster response times but potentially less comprehensive answers
- **Multi API** provides more thorough analysis but takes longer to process
- **Large embeddings** generally provide better quality but slower response times
- **Small embeddings** offer faster processing suitable for real-time applications

## Technical Notes

- All tests conducted with 3-second delays between requests
- Timeout set to 120 seconds per request
- Quality scoring based on keyword matching, answer structure, and completeness
- Results saved with timestamp 20250616_195319 for reproducibility

---
*Generated by RAG Benchmark System for Thesis Research*
