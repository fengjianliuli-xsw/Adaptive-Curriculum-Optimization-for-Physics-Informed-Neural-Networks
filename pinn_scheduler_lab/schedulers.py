"""No controller network; policies depend only on measured training/probe state."""
import numpy as np

PLANS={'A':'始终 Joint','B':'BC → PDE → Joint','C':'PDE → BC → Joint','D':'ReLoBRaLo','E':'规则自适应切换'}


class Controller:
    def __init__(self,plan,cfg,seed):
        self.plan,self.cfg=plan,cfg
        self.action='Joint';self.weights=np.ones(2)
        self.rng=np.random.default_rng(seed+900)
        self.initial=None;self.previous=None;self.changed_check=-10000;self.check=0

    def decision(self,state,fraction):
        self.check+=1
        if self.plan in ('A','D'):return 'Joint','joint_baseline'
        if self.plan in ('B','C'):
            actions=['BC','PDE','Joint'] if self.plan=='B' else ['PDE','BC','Joint']
            return actions[min(2,int(fraction*3))],'fixed_equal_time_thirds'
        if self.check-self.changed_check<self.cfg.min_dwell_checks:return self.action,'minimum_dwell'
        bc,pde,cos=state['e_bc'],state['e_pde'],state['gradient_cosine']
        high,low=self.cfg.tau_bc*(1+self.cfg.hysteresis),self.cfg.tau_bc*(1-self.cfg.hysteresis)
        if bc>high:
            action='BC' if bc>2*self.cfg.tau_bc or self.action=='BC' else 'Joint';reason='boundary_violation'
        elif pde<=self.cfg.tau_pde:action,reason='Joint','both_ready_for_refinement'
        elif bc<=low:
            action,reason=('Joint','conflict_and_stagnation') if cos<-.5 and state['relative_pde_drop']<.001 else ('PDE','boundary_safe_pde_unresolved')
        else:action,reason='Joint','boundary_hysteresis_band'
        if self.action=='BC' and action=='BC' and state['relative_bc_drop']<.001 and bc<2*self.cfg.tau_bc:
            action,reason='Joint','boundary_stagnation_escape'
        if action!=self.action:self.changed_check=self.check
        return action,reason

    def reweight(self,losses):
        losses=np.asarray(losses,dtype=float)
        if self.initial is None:
            self.initial=losses.copy();self.previous=losses.copy();return self.weights.copy()
        def soft(reference):
            score=losses/(np.maximum(reference,1e-12)*self.cfg.relobralo_temperature)
            score-=score.max();value=np.exp(score);return 2*value/value.sum()
        rho=float(self.rng.random()<self.cfg.relobralo_beta)
        alpha=self.cfg.relobralo_alpha
        self.weights=alpha*(rho*self.weights+(1-rho)*soft(self.initial))+(1-alpha)*soft(self.previous)
        self.previous=losses.copy();return self.weights.copy()
