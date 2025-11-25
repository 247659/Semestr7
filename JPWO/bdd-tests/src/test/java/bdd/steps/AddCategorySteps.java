package bdd.steps;

import io.cucumber.java.en.Then;
import io.cucumber.java.en.When;

import static bdd.steps.SharedClass.addCategoryPage;
import static bdd.steps.SharedClass.categoryAdminPage;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class AddCategorySteps {

    private String uniqueCategoryName;
    private String uniqueSlug;

    @When("I open the add category page")
    public void iOpenTheAddCategoryPage() {
        addCategoryPage.open();
    }

    @When("I create a new category")
    public void iCreateANewCategory() {
        uniqueCategoryName = "New Category " + System.currentTimeMillis();
        uniqueSlug = "new-category-" + System.currentTimeMillis();

        addCategoryPage.fillForm(uniqueCategoryName, uniqueSlug);
    }

    @Then("the category should be visible in the admin categories list")
    public void theCategoryShouldBeVisibleInTheAdminCategoriesList() {
        categoryAdminPage.open();
        categoryAdminPage.search(uniqueCategoryName);

        String addedName = categoryAdminPage.getCategoryName(uniqueCategoryName);
        String addedSlug = categoryAdminPage.getCategorySlug(uniqueCategoryName);

        assertEquals(uniqueCategoryName, addedName, "Category name does not match");
        assertEquals(uniqueSlug, addedSlug, "Category slug does not match");
    }
}
