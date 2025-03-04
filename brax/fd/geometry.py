import jax.numpy as jnp

def skew_to_vector(self, skew_matrix):
    """Convert a 3x3 skew-symmetric matrix to a 3D vector (vee operator)."""
    return jnp.array([skew_matrix[2, 1], skew_matrix[0, 2], skew_matrix[1, 0]])

def matrix_log(self, R):
    def compute_log(R):
        # Compute the angle of rotation
        theta = jnp.arccos((jnp.trace(R) - 1) / 2)
        return (theta / (2 * jnp.sin(theta))) * (R - R.T)  # Normal log formula otherwise

    # Handle the special case where theta is very small (close to zero)
    # Instead of a conditional, use jnp.where to return a zero matrix when theta is very small
    return jnp.where(
        jnp.isclose((jnp.trace(R) - 1) / 2, 1.),
        jnp.zeros((3, 3)),  # Return zero matrix if theta is close to 0
        compute_log(R)
    )

def rotation_distance(RA, RB):
    """Compute the geodesic distance between two SO(3) rotation matrices."""
    relative_rotation = jnp.dot(RA.T, RB)  # Compute RA^T * RB
    log_relative = matrix_log(relative_rotation)  # Compute matrix logarithm
    omega = skew_to_vector(log_relative)    # Extract the rotation vector (vee operator)
    return jnp.linalg.norm(omega)           # Compute the rotation distance
