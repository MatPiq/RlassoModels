:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:

   {% block methods %}
   .. automethod:: __init__
   .. automethod:: fit
   .. automethod:: fit_formula
   .. automethod:: predict

   {% endblock %}

.. include:: {{module}}.{{objname}}.examples

.. raw:: html

    <div style='clear:both'></div>
