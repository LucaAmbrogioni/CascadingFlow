

def variational_update(prior_model, variational_model, data, optimizer, batch_size):
    # Gradient reset
    optimizer.zero_grad()

    # Variational loss
    X, mu, x_pre, log_jacobian, epsilon_loss = variational_model.sample_timeseries(batch_size, data)
    log_q = variational_model.evaluate_avg_joint_log_prob(X, None, mu, x_pre=x_pre, log_jacobian=log_jacobian,
                                                          epsilon_loss=epsilon_loss)
    log_p = prior_model.evaluate_avg_joint_log_prob(X, data, mu)
    loss = (log_q - log_p)

    # Update
    loss.backward()
    optimizer.step()
    return loss

def variational_img_update(prior_model, variational_model, data, bin_list, optimizer, batch_size):
    # Gradient reset
    optimizer.zero_grad()

    # Variational loss
    X, mu, x_pre, log_jacobian, epsilon_loss = variational_model.sample_timeseries(batch_size, data)
    log_q = variational_model.evaluate_avg_joint_log_prob(X, None, mu, bin_list, x_pre=x_pre, log_jacobian=log_jacobian,
                                                          epsilon_loss=epsilon_loss)
    log_p = prior_model.evaluate_avg_joint_log_prob(X, data, mu, bin_list)
    loss = (log_q - log_p)

    # Update
    loss.backward()
    optimizer.step()
    return loss