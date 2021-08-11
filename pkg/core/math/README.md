# Native vector/matrix/tensor implementation

Calculations are performed inplace as much as possible to reduce allocations and copying.

Most places assume that vector/matrix sizes are correct. Other places panic on wrong sizes. 