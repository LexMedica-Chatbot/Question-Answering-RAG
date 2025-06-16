# Comprehensive RAG Benchmark Report

**Generated:** 2025-06-16T13:17:08.481239  
**Timestamp:** 20250616_131708

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
- Average Response Time: 15906 ms
- Median Response Time: 13914 ms
- Response Time Range: 5935 - 33166 ms

**Answer Quality:**
- Average Answer Length: 1794 characters
- Average Word Count: 241.4 words
- Keyword Matching Score: 0.924
- Structure Score: 0.800
- Completeness Score: 0.952
- **Overall Quality Score: 0.892**

**Technical Metrics:**
- Average Documents Retrieved: 4.0
- Average Processing Steps: 1.0

**Performance by Difficulty:**

- **Easy**: 100.0% success, 7879ms avg, 0.798 quality
- **Medium**: 100.0% success, 14227ms avg, 0.844 quality
- **Hard**: 100.0% success, 18621ms avg, 0.978 quality
- **Very_Hard**: 100.0% success, 27230ms avg, 1.000 quality


### SIMPLE API + LARGE Embedding

**Overall Performance:**
- Success Rate: 100.0%
- Average Response Time: 14318 ms
- Median Response Time: 12637 ms
- Response Time Range: 5285 - 37281 ms

**Answer Quality:**
- Average Answer Length: 2214 characters
- Average Word Count: 300.8 words
- Keyword Matching Score: 0.944
- Structure Score: 0.783
- Completeness Score: 0.963
- **Overall Quality Score: 0.897**

**Technical Metrics:**
- Average Documents Retrieved: 4.0
- Average Processing Steps: 1.0

**Performance by Difficulty:**

- **Easy**: 100.0% success, 7631ms avg, 0.798 quality
- **Medium**: 100.0% success, 12866ms avg, 0.858 quality
- **Hard**: 100.0% success, 13825ms avg, 0.978 quality
- **Very_Hard**: 100.0% success, 27990ms avg, 1.000 quality


### MULTI API + SMALL Embedding

**Overall Performance:**
- Success Rate: 100.0%
- Average Response Time: 64129 ms
- Median Response Time: 70034 ms
- Response Time Range: 13767 - 98763 ms

**Answer Quality:**
- Average Answer Length: 1188 characters
- Average Word Count: 156.2 words
- Keyword Matching Score: 0.653
- Structure Score: 0.450
- Completeness Score: 0.695
- **Overall Quality Score: 0.599**

**Technical Metrics:**
- Average Documents Retrieved: 5.4
- Average Processing Steps: 3.8

**Performance by Difficulty:**

- **Easy**: 100.0% success, 21919ms avg, 0.701 quality
- **Medium**: 100.0% success, 74293ms avg, 0.565 quality
- **Hard**: 100.0% success, 78708ms avg, 0.626 quality
- **Very_Hard**: 100.0% success, 85248ms avg, 0.473 quality


### MULTI API + LARGE Embedding

**Overall Performance:**
- Success Rate: 100.0%
- Average Response Time: 70535 ms
- Median Response Time: 74872 ms
- Response Time Range: 2924 - 118600 ms

**Answer Quality:**
- Average Answer Length: 1424 characters
- Average Word Count: 188.8 words
- Keyword Matching Score: 0.597
- Structure Score: 0.517
- Completeness Score: 0.683
- **Overall Quality Score: 0.599**

**Technical Metrics:**
- Average Documents Retrieved: 6.1
- Average Processing Steps: 3.8

**Performance by Difficulty:**

- **Easy**: 100.0% success, 26996ms avg, 0.767 quality
- **Medium**: 100.0% success, 69880ms avg, 0.499 quality
- **Hard**: 100.0% success, 86430ms avg, 0.626 quality
- **Very_Hard**: 100.0% success, 113308ms avg, 0.506 quality


## Comparative Analysis

### Key Findings

**🏆 Best Overall Performance:** Simple + Large
- Combined score (success rate × quality): 89.68

**⚡ Fastest Response:** Simple + Large
- Average response time: 14318 ms

**🎯 Highest Quality:** Simple + Large
- Average quality score: 0.897

**📚 Most Comprehensive:** Multi + Large
- Average documents retrieved: 6.1

## Recommendations

### For Production Use:
1. **High-Performance Requirements:** Use Simple + Large
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
- Results saved with timestamp 20250616_131708 for reproducibility

---
*Generated by RAG Benchmark System for Thesis Research*
