// LotF C++ Extension Module
// Implements diversity-aware search result filtering using cutoff tables

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

// Use boost flat containers if available for better performance
#if __has_include(<boost/unordered/unordered_flat_map.hpp>)
  #include <boost/unordered/unordered_flat_map.hpp>
  #include <boost/unordered/unordered_flat_set.hpp>
  #define USE_BOOST_FLAT_CONTAINERS
#else
  #include <unordered_map>
  #include <unordered_set>
#endif

namespace nb = nanobind;

using namespace nb::literals;

/**
 * @brief Returns the type of unordered container implementation being used
 * 
 * Useful for debugging and performance analysis
 * 
 * @return String describing the container type (boost or std)
 */
const char* get_unordered_container_type() {
#ifdef USE_BOOST_FLAT_CONTAINERS
    return "boost::unordered_flat_map/set";
#else
    return "std::unordered_map/set";
#endif
}

/**
 * @brief Core diversity filtering algorithm
 * 
 * Filters candidate search results to produce diverse final results using cutoff tables.
 * Implements the diversification algorithm from the paper using cutoff tables to remove
 * similar items from search results.
 * 
 * @param flattened_neighbors Flattened cutoff table entries
 * @param neighbor_offsets Cumulative sum of cutoff table lengths
 * @param dists Candidate distances (Nq x candidate_k)
 * @param ids Candidate IDs (Nq x candidate_k)
 * @param diverse_dists Output diverse distances (Nq x final_k)
 * @param diverse_ids Output diverse IDs (Nq x final_k)
 */
void search_cpp(
    nb::ndarray<const int64_t, nb::ndim<1>, nb::device::cpu> flattened_neighbors,
    nb::ndarray<const int64_t, nb::ndim<1>, nb::device::cpu> neighbor_offsets,
    nb::ndarray<const float, nb::ndim<2>, nb::device::cpu> dists,
    nb::ndarray<const int64_t, nb::ndim<2>, nb::device::cpu> ids,
    nb::ndarray<float, nb::ndim<2>, nb::device::cpu> diverse_dists,
    nb::ndarray<int64_t, nb::ndim<2>, nb::device::cpu> diverse_ids
    )
{
    // Get array views for efficient access
    auto v_flattened_neighbors = flattened_neighbors.view();
    auto v_neighbor_offsets = neighbor_offsets.view();
    auto v_dists = dists.view();
    auto v_ids = ids.view();
    auto v_diverse_dists = diverse_dists.view();
    auto v_diverse_ids = diverse_ids.view();
    
    // Extract dimensions
    size_t Nq = v_dists.shape(0);              // number of queries
    size_t candidate_k = v_dists.shape(1);     // number of candidates per query
    size_t final_k = v_diverse_dists.shape(1);  // number of final results per query

    // Validate array dimensions
    assert(final_k <= candidate_k);
    assert(v_ids.shape(0) == Nq);
    assert(v_ids.shape(1) == candidate_k);
    assert(v_diverse_dists.shape(1) == final_k);
    assert(v_diverse_ids.shape(0) == Nq);
    assert(v_diverse_ids.shape(1) == final_k);

    // Process each query independently
    // In the paper, we use a data structure called OrderedSet. In the actual code,
    // we use an inlined version of its contents instead, as this was slightly faster.
    // The processing is the same as when using OrderedSet.
    for (size_t nq = 0; nq < Nq; ++nq) {
        // Initialize containers for tracking available items and their distances
#ifdef USE_BOOST_FLAT_CONTAINERS
        boost::unordered_flat_set<int64_t> items_set;  // "V" in the paper
        boost::unordered_flat_map<int64_t, float> dist_map;  // The mapping from an ID to its distance
#else
        std::unordered_set<int64_t> items_set;
        std::unordered_map<int64_t, float> dist_map;
#endif
        items_set.reserve(candidate_k);
        dist_map.reserve(candidate_k);

        // Initialize available items set and distance lookup
        for (size_t k = 0; k < candidate_k; ++k) {
            int64_t id = v_ids(nq, k);
            dist_map[id] = v_dists(nq, k);
            items_set.insert(id);
        }
 
        size_t ids_final_idx = 0;  // index into final results array
        size_t head = 0;            // counter. "c" in the paper

        // Main diversification loop
        while (ids_final_idx < final_k) {  // L3 in Alg 2
            // Pop. L4 in Alg 2
            int64_t k = -1;
            for (size_t n = head; n < candidate_k; ++n) {
                k = v_ids(nq, n);
                if (items_set.find(k) != items_set.end()) {
                    items_set.erase(k);  
                    head = n + 1;        
                    break;
                }
            }
                        
            // L5 in Alg 2. Add selected item to final results
            v_diverse_ids(nq, ids_final_idx) = k;
            ids_final_idx++;

            // Apply cutoff table: remove items that are too similar to selected item
            for (int64_t j = v_neighbor_offsets(k); j < v_neighbor_offsets(k + 1); ++j) {
                int64_t i = v_flattened_neighbors(j);

                // Safe guard (Sec 4.5): if remaining items exactly fill final_k, use them all
                if (ids_final_idx + items_set.size() == final_k) {
                    size_t z = ids_final_idx;
                    for (size_t k = 0; k < candidate_k; ++k) {
                        if ( items_set.find(v_ids(nq, k)) != items_set.end() ) {
                            v_diverse_ids(nq, z) = v_ids(nq, k);
                            z++;
                        }
                    }

                    ids_final_idx = final_k;
                    break;
                }
                // L6 in Alg 2.Remove similar item from available set
                items_set.erase(i);
            }
 

        }

        // Fill in corresponding distances for final results
        for (size_t i = 0; i < final_k; ++i) {
            v_diverse_dists(nq, i) = dist_map[v_diverse_ids(nq, i)];          
        }

    }
}




// Python module definition
NB_MODULE(lotf_ext, m) {
    m.doc() = "lotf c++ extension module";
    
    // Utility function to check container implementation
    m.def("get_unordered_container_type", &get_unordered_container_type, 
        "Returns the type of unordered container implementation being used");
    
    // Main diversity filtering function
    m.def("search_cpp", &search_cpp, 
        "flattened_neighbors"_a,   // flattened cutoff table
        "neighbor_offsets"_a,      // cumulative lengths
        "dists"_a,                // candidate distances
        "ids"_a,                  // candidate IDs
        "diverse_dists"_a,     // output distances
        "diverse_ids"_a,       // output IDs
        "Performs diversity-aware filtering of search results using cutoff tables"
        );
}
