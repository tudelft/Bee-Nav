# Run MNIST experimetns
import mushroom_body.MNIST_MB as mnist

print("--------------------------------------------------")
print("Training different network types on MNIST dataset:")
print("--------------------------------------------------")

print("***** Training MBON network: *****")
print(f"20000 Kenyon Cells")
mnist.train_MNIST(network_type='MBON', KC_dimension=20000, n_nonzero_weights = 10)
print(f"5000 Kenyon Cells")
mnist.train_MNIST(network_type='MBON', KC_dimension=5000, n_nonzero_weights = 4)
print("***** Training attention network: *****")
mnist.train_MNIST(network_type='attention')
print("***** Training compact network: *****")
mnist.train_MNIST(network_type='compact')
print("***** Training simple network: *****")
mnist.train_MNIST(network_type='simple')

