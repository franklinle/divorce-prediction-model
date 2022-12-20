$(document).ready(function() {
    $('form-row').submit(function(event) {
      event.preventDefault();  // Prevent the form from being submitted
  
      // Get the value of the data-result attribute
      var result = $('body').data('prediction_result');
  
      // Update the background based on the value of the result
      if (result > 0.8) {
        $('body').css('background-image', 'url(../images/heart.jpeg)');
      } else if (result < 0.2) {
        $('body').css('background-image', 'url(../images/heart.jpeg)');
      }
    });
  });
  