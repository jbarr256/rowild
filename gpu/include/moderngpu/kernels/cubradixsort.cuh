#include "device/deviceutil.cuh"
#include "hipcub/hipcub.hpp"

namespace mgpu {

template<typename Key>
bool CubRadixSort(Key* keys_global, Key* keys2_global, int count, int beginBit,
	int endBit, CudaContext& context) {

	hipcub::DoubleBuffer<Key> keys(keys_global, keys2_global);

	size_t tempBytes = 0;
	hipcub::DeviceRadixSort::SortKeys(0, tempBytes, keys, count, beginBit, endBit,
		context.Stream());

	MGPU_MEM(byte) tempDevice = context.Malloc<byte>(tempBytes);

	hipcub::DeviceRadixSort::SortKeys(tempDevice->get(), tempBytes, keys, count,
		beginBit, endBit, context.Stream());
	MGPU_SYNC_CHECK("hipcub::DeviceRadixSort::SortKeys");

	return 1 == keys.selector;
}

template<typename Key, typename Value>
bool CubRadixSort(Key* keys_global, Key* keys2_global, Value* values_global,
	Value* values2_global, int count, int beginBit, int endBit,
	CudaContext& context) {

	hipcub::DoubleBuffer<Key> keys(keys_global, keys2_global);
	hipcub::DoubleBuffer<Value> values(values_global, values2_global);

	size_t tempBytes = 0;
	hipcub::DeviceRadixSort::SortPairs(0, tempBytes, keys, values, count,
		beginBit, endBit, context.Stream());

	MGPU_MEM(byte) tempDevice = context.Malloc<byte>(tempBytes);

	hipcub::DeviceRadixSort::SortPairs(tempDevice->get(), tempBytes, keys, values, 
		count, beginBit, endBit, context.Stream());
	MGPU_SYNC_CHECK("hipcub::DeviceRadixSort::SortPairs");

	return 1 == keys.selector;
}

} // namespace mgpu
