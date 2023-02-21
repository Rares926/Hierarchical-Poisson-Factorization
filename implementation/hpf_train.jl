using NPZ
using Gen
using Plots # julia package used for plotting
using LinearAlgebra

xs = npzread("Z:/Master I/PP    - Probabilistic Programming/Project/input/input_10x10.npy");

num_users = 10
num_items = 10
k = 10 # pentru primele 10 filme sunt 10 categorii
a, c, a_prim, b_prim, c_prim, d_prim = [0.3 for _ = 1:6];
n_burnin = 100000
n_samples = 40000

@gen function hpf_model(num_users::Int64,
                        num_items::Int64,
                        k::Int64,
                        a::Float64, c::Float64,
                        a_prim::Float64, b_prim::Float64, c_prim::Float64, d_prim::Float64)
    
    # for each user u
    X_preference = Vector[] # asta i theta care trebuie resimulat practic 
    for u = 1:num_users
        activity = ({(:activity,u)} ~ gamma(a_prim,a_prim/b_prim))
        preference = Float64[]
        for _k = 1:k  # for each component k ( k is like a random number of categories)
            push!(preference,{(:preference,u, _k)} ~ gamma(a,activity))
        end
        push!(X_preference,preference)
    end

    # for each item i 
    X_sample_attribute = Vector[] # asta i beta care trebuie resimulat
    for i = 1:num_items
        popularity = ({(:popularity,i)} ~ gamma(c_prim,c_prim/d_prim))
        attribute = Float64[]
        for _k = 1:k # for each component k ( k is like a random number of categories)
            push!(attribute,{(:attribute, i, _k)} ~ gamma(c,popularity))
        end
        push!(X_sample_attribute, attribute)
    end
    
    # for each user u and item i
    y = Matrix{Float64}(undef,0,num_items)
    for u = 1:num_users
        rating = Float64[]
        for i = 1:num_items
            push!(rating,{(:rating, u, i)} ~ poisson(dot(X_preference[u],X_sample_attribute[i])))
        end
        y = vcat(y, rating')
    end
    
    y
end

function make_constraints(ys::Matrix{Float64})
    constraints = Gen.choicemap()
    for u=1:size(ys)[1]
        for i=1:size(ys)[2]  
        constraints[(:rating, u, i)] = ys[u,i]
        end
    end
    constraints
end;

function block_resimulation_update(tr,num_users, num_items, k)

    # Block 1: Update preference (theta)
    for u = 1:num_users
        for _k = 1:k 
        latent_variable = select(:preference,u,_k)
        (tr, _) = mh(tr, latent_variable)
        end
    end


    # Block 2: Update attribute (beta)
    for i = 1:num_items
        for _k = 1:k
        latent_variable = select(:attribute,i,_k)
        (tr, _) = mh(tr, latent_variable)
        end
    end

    tr

end

function block_resimulation_inference(n_burnin, n_samples, thin)
    observations = make_constraints(xs)
    (tr, _) = generate(hpf_model, (num_users, num_items, k, a, c, a_prim, b_prim, c_prim, d_prim), observations)
    for iter=1:n_burnin
        tr = block_resimulation_update(tr,num_users,num_items,k)
        if iter % 100 == 0 
            print(iter)
        end

    end
    trs = []
    for iter=1:n_samples
        for itert = 1:thin # reduce the autocorrelation in a sample of generated data by MCMC
            tr = block_resimulation_update(tr,num_users,num_items,k)
        end
        push!(trs, tr)
        if iter % 100 == 0 
            print(iter)
        end
    end

    trs

end;

function create_preference_vector(one_trace,num_users,num_items,k)
    X_preference = Vector[] # asta i theta care trebuie resimulat practic 
    for u = 1:num_users
        preference = Float64[]
        for _k = 1:k  
            push!(preference,one_trace[(:preference,u,_k)])
        end

        push!(X_preference,preference)
    end

    X_sample_attribute = Vector[] # asta i beta care trebuie resimulat
    for i = 1:num_items
        attribute = Float64[]
        for _k = 1:k 
            push!(attribute,one_trace[(:attribute, i, _k)])
        end
        push!(X_sample_attribute, attribute)
    end

    y = Matrix{Float64}(undef,0,num_items)
    for u = 1:num_users
        rating = Float64[]
        for i = 1:num_items
            push!(rating, poisson(dot(X_preference[u]',X_sample_attribute[i])))
        end
        y = vcat(y, rating)
    end
     
    y

end

trs = block_resimulation_inference(100000, 40000, 2);
final_matrix = zeros(num_users, num_items)
for i=1:n_samples
    y = create_preference_vector(trs[i],num_users,num_items,k)
    final_matrix = final_matrix + y
end
println(final_matrix/n_samples)

