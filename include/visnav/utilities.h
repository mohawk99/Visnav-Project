#include <visnav/common_types.h>
#include <iostream>
#include <set>
#include <algorithm>

template <typename T>
std::set<T, std::greater<T>> sortSetDescending(const std::set<T>& inputSet) {
  std::set<T, std::greater<T>> sortedSet(inputSet.begin(), inputSet.end());
  return sortedSet;
}

template <typename T>
std::set<T, std::greater<T>> getTopNElements(const std::set<T>& inputSet,
                                             int N) {
  std::set<T, std::greater<T>> sortedSet = sortSetDescending(inputSet);

  // Extract the top N elements
  auto it = sortedSet.begin();
  std::advance(it, N);

  std::set<T, std::greater<T>> topNSet(sortedSet.begin(), it);
  return topNSet;
}
