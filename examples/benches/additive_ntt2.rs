// Copyright 2024-2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_field::{
	AESTowerField32b, AESTowerField128b, BinaryField32b, BinaryField128b, BinaryField128bPolyval,
	PackedExtension, TowerField, arch::OptimalUnderlier, as_packed_field::PackedType,
};
use binius_ntt::{AdditiveNTT, NTTShape, SingleThreadedNTT};

fn bench_large_transform<F: TowerField, PE: PackedExtension<F>>(field: &str) {
	let log_dim = 20;

	let data_len = 1 << (log_dim - PE::LOG_WIDTH);
	let mut rng = rand::rng();
	let mut data = repeat_with(|| PE::random(&mut rng))
		.take(data_len)
		.collect::<Vec<_>>();

	// let params = format!("{field}");
	// group.throughput(Throughput::Bytes((data_len * size_of::<PE>()) as u64));

	let shape = NTTShape {
		log_y: log_dim,
		..Default::default()
	};

	let ntt = SingleThreadedNTT::<F>::new(log_dim)
		.unwrap()
		.precompute_twiddles()
		.multithreaded();

	// group.bench_function(BenchmarkId::new("multithread/precompute", &params), |b| {
	// 	b.iter(|| ntt.forward_transform_ext(&mut data, shape, 0, 0, 0));
	// });

	println!("yoooooo2");

	let _ = ntt.forward_transform_ext(&mut data, shape, 0, 0, 0);

	// println!("yeeet");
}

fn main() {
	bench_large_transform::<BinaryField32b, PackedType<OptimalUnderlier, BinaryField128b>>(
		"field=BinaryField128b",
	);

	// bench_large_transform::<AESTowerField32b, PackedType<OptimalUnderlier, AESTowerField128b>>(
	// 	"field=AESTowerField128b",
	// );

	// bench_large_transform::<
	// 	BinaryField128bPolyval,
	// 	PackedType<OptimalUnderlier, BinaryField128bPolyval>,
	// >("field=BinaryField128bPolyval");
}
